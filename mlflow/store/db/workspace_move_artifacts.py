from __future__ import annotations

import os
import posixpath
import tempfile
from dataclasses import dataclass

import sqlalchemy as sa

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.model_registry.dbmodels.models import SqlModelVersion
from mlflow.store.tracking.dbmodels.models import (
    SqlExperiment,
    SqlLoggedModel,
    SqlRun,
    SqlTraceInfo,
    SqlTraceTag,
)
from mlflow.store.workspace.dbmodels.models import SqlWorkspace
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.workspace_utils import WORKSPACES_DIR_NAME

# Artifact URIs that only a running tracking server can resolve. A database-side
# command has no server to proxy through, so relocation is limited to roots the
# artifact repositories can reach directly (file, s3, and so on).
_SERVER_RESOLVED_URI_PREFIXES = ("mlflow-artifacts:", "http://", "https://")

_SKIPPED_URI_SAMPLE_SIZE = 5


@dataclass(frozen=True)
class ExperimentArtifactPlan:
    """Relocation plan for one experiment's artifacts."""

    experiment_id: int
    experiment_name: str
    old_root: str
    new_root: str
    run_uri_count: int
    model_uri_count: int
    trace_uri_count: int
    # Stored URIs under the experiment that are not under old_root. They are
    # reported and left untouched.
    skipped_uri_count: int = 0
    skipped_uri_sample: tuple[str, ...] = ()
    # model_versions.storage_location rows pointing into the old root. Registry
    # rows are never rewritten by an experiments move. These are surfaced so the
    # administrator knows those references keep pointing at the old prefix.
    registry_reference_count: int = 0


def _is_under_root(uri: str | None, root: str) -> bool:
    if not uri:
        return False
    return uri == root or uri.startswith(root.rstrip("/") + "/")


def _rewritten_uri(uri: str, old_root: str, new_root: str) -> str:
    if uri == old_root:
        return new_root
    suffix = uri[len(old_root.rstrip("/")) :]
    return new_root.rstrip("/") + suffix


def _ensure_directly_accessible(uri: str, description: str) -> None:
    if uri.startswith(_SERVER_RESOLVED_URI_PREFIXES):
        raise RuntimeError(
            f"{description} {uri!r} can only be resolved through a running tracking server. "
            "Artifact relocation supports directly accessible artifact roots "
            "(e.g. file://, s3://) only."
        )


def _resolve_target_root_base(
    conn, target_workspace: str, default_artifact_root: str | None
) -> str:
    """Mirror the workspace provider's artifact root resolution for the target workspace.

    A workspace-level default_artifact_root is used as is, matching the provider
    returning append_workspace_prefix=False for it. The server-level root gets the
    workspaces/<name> suffix appended, matching the provider default.
    """
    workspaces = SqlWorkspace.__table__
    workspace_root = conn.execute(
        sa.select(workspaces.c.default_artifact_root).where(workspaces.c.name == target_workspace)
    ).scalar()
    if workspace_root:
        return workspace_root
    if not default_artifact_root:
        raise RuntimeError(
            f"Cannot determine the artifact root for workspace {target_workspace!r}: the "
            "workspace has no default_artifact_root configured. Pass --default-artifact-root "
            "with the same value the tracking server is started with."
        )
    return append_to_uri_path(default_artifact_root, WORKSPACES_DIR_NAME, target_workspace)


def _like_prefix_pattern(prefix: str) -> str:
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "/%"


def build_experiment_artifact_plans(
    conn,
    experiment_names: list[str],
    source_workspace: str,
    target_workspace: str,
    default_artifact_root: str | None,
) -> list[ExperimentArtifactPlan]:
    experiments = SqlExperiment.__table__
    target_base = _resolve_target_root_base(conn, target_workspace, default_artifact_root)
    _ensure_directly_accessible(target_base, "Target artifact root")

    rows = conn.execute(
        sa
        .select(
            experiments.c.experiment_id,
            experiments.c.name,
            experiments.c.artifact_location,
        )
        .where(
            experiments.c.workspace == source_workspace,
            experiments.c.name.in_(experiment_names),
        )
        .order_by(experiments.c.name)
    ).fetchall()

    plans = []
    for experiment_id, name, old_root in rows:
        if not old_root:
            raise RuntimeError(
                f"Experiment {name!r} has no artifact_location, cannot relocate its artifacts."
            )
        _ensure_directly_accessible(old_root, f"Artifact location of experiment {name!r}")
        new_root = append_to_uri_path(target_base, str(experiment_id))
        if _is_under_root(new_root, old_root) or _is_under_root(old_root, new_root):
            raise RuntimeError(
                f"Source and target artifact roots overlap for experiment {name!r}: "
                f"{old_root!r} vs {new_root!r}."
            )
        plans.append(_plan_for_experiment(conn, experiment_id, name, old_root, new_root))
    return plans


def _plan_for_experiment(conn, experiment_id, name, old_root, new_root) -> ExperimentArtifactPlan:
    runs = SqlRun.__table__
    models = SqlLoggedModel.__table__
    trace_info = SqlTraceInfo.__table__
    trace_tags = SqlTraceTag.__table__
    model_versions = SqlModelVersion.__table__

    skipped_count = 0
    skipped_sample: list[str] = []

    def _count_under_root(uris) -> int:
        nonlocal skipped_count
        matched = 0
        for uri in uris:
            if _is_under_root(uri, old_root):
                matched += 1
            elif uri:
                skipped_count += 1
                if len(skipped_sample) < _SKIPPED_URI_SAMPLE_SIZE:
                    skipped_sample.append(uri)
        return matched

    run_uris = conn.execute(
        sa.select(runs.c.artifact_uri).where(runs.c.experiment_id == experiment_id)
    ).scalars()
    model_uris = conn.execute(
        sa.select(models.c.artifact_location).where(models.c.experiment_id == experiment_id)
    ).scalars()
    trace_request_ids = sa.select(trace_info.c.request_id).where(
        trace_info.c.experiment_id == experiment_id
    )
    trace_uris = conn.execute(
        sa.select(trace_tags.c.value).where(
            trace_tags.c.request_id.in_(trace_request_ids),
            trace_tags.c.key == MLFLOW_ARTIFACT_LOCATION,
        )
    ).scalars()
    registry_reference_count = conn.execute(
        sa
        .select(sa.func.count())
        .where(
            SqlModelVersion.__table__.c.storage_location.like(
                _like_prefix_pattern(old_root), escape="\\"
            )
        )
        .select_from(model_versions)
    ).scalar()

    return ExperimentArtifactPlan(
        experiment_id=experiment_id,
        experiment_name=name,
        old_root=old_root,
        new_root=new_root,
        run_uri_count=_count_under_root(run_uris),
        model_uri_count=_count_under_root(model_uris),
        trace_uri_count=_count_under_root(trace_uris),
        skipped_uri_count=skipped_count,
        skipped_uri_sample=tuple(skipped_sample),
        registry_reference_count=registry_reference_count,
    )


def _walk_files(repo):
    stack: list[str | None] = [None]
    while stack:
        subdir = stack.pop()
        for info in repo.list_artifacts(subdir):
            if info.is_dir:
                stack.append(info.path)
            else:
                yield info


def copy_experiment_artifacts(plan: ExperimentArtifactPlan) -> int:
    """Copy every artifact under plan.old_root to plan.new_root and verify the copy.

    Uploads overwrite, so a failed run can simply be rerun. Source objects are never
    modified or deleted. Returns the number of files copied.
    """
    src_repo = get_artifact_repository(plan.old_root)
    dst_repo = get_artifact_repository(plan.new_root)
    copied = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        for info in _walk_files(src_repo):
            local_path = src_repo.download_artifacts(info.path, tmp_dir)
            dst_repo.log_artifact(local_path, posixpath.dirname(info.path) or None)
            # Remove each staged file once uploaded so peak temp disk usage is
            # bounded by the largest artifact, not the whole experiment.
            os.remove(local_path)
            copied += 1
    _verify_copy(src_repo, dst_repo, plan)
    return copied


def _verify_copy(src_repo, dst_repo, plan: ExperimentArtifactPlan) -> None:
    src_files = {(info.path, info.file_size) for info in _walk_files(src_repo)}
    dst_files = {(info.path, info.file_size) for info in _walk_files(dst_repo)}
    if missing := src_files - dst_files:
        sample = ", ".join(path for path, _ in sorted(missing)[:3])
        raise RuntimeError(
            f"Artifact copy verification failed for experiment {plan.experiment_name!r}: "
            f"{len(missing)} file(s) missing at {plan.new_root!r} (e.g. {sample}). "
            "The database was not modified. Rerun the command to retry."
        )


def rewrite_experiment_artifact_uris(conn, plan: ExperimentArtifactPlan) -> None:
    """Rewrite stored artifact URIs under plan.old_root to plan.new_root.

    Runs inside the caller's move transaction so the workspace flip and the URI
    rewrites commit atomically. URIs not under the old root are left untouched.
    """
    experiments = SqlExperiment.__table__
    runs = SqlRun.__table__
    models = SqlLoggedModel.__table__
    trace_info = SqlTraceInfo.__table__
    trace_tags = SqlTraceTag.__table__

    conn.execute(
        experiments
        .update()
        .where(experiments.c.experiment_id == plan.experiment_id)
        .values(artifact_location=plan.new_root)
    )
    _rewrite_column(
        conn,
        runs,
        pk_cols=(runs.c.run_uuid,),
        uri_col=runs.c.artifact_uri,
        row_filter=runs.c.experiment_id == plan.experiment_id,
        plan=plan,
    )
    _rewrite_column(
        conn,
        models,
        pk_cols=(models.c.model_id,),
        uri_col=models.c.artifact_location,
        row_filter=models.c.experiment_id == plan.experiment_id,
        plan=plan,
    )
    trace_request_ids = sa.select(trace_info.c.request_id).where(
        trace_info.c.experiment_id == plan.experiment_id
    )
    _rewrite_column(
        conn,
        trace_tags,
        pk_cols=(trace_tags.c.request_id, trace_tags.c.key),
        uri_col=trace_tags.c.value,
        row_filter=sa.and_(
            trace_tags.c.request_id.in_(trace_request_ids),
            trace_tags.c.key == MLFLOW_ARTIFACT_LOCATION,
        ),
        plan=plan,
    )


def _rewrite_column(conn, table, pk_cols, uri_col, row_filter, plan) -> None:
    rows = conn.execute(sa.select(*pk_cols, uri_col).where(row_filter)).fetchall()
    updates = [
        {
            **{f"pk_{i}": row[i] for i in range(len(pk_cols))},
            "new_uri": _rewritten_uri(row[-1], plan.old_root, plan.new_root),
        }
        for row in rows
        if _is_under_root(row[-1], plan.old_root)
    ]
    if not updates:
        return
    where_clause = sa.and_(*[col == sa.bindparam(f"pk_{i}") for i, col in enumerate(pk_cols)])
    stmt = table.update().where(where_clause).values({uri_col.name: sa.bindparam("new_uri")})
    conn.execute(stmt, updates)
