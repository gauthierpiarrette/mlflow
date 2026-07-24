from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa

from mlflow.store.db.workspace_move_artifacts import (
    ExperimentArtifactPlan,
    build_experiment_artifact_plans,
    copy_experiment_artifacts,
    rewrite_experiment_artifact_uris,
)
from mlflow.store.db.workspace_utils import (
    MODEL_CHILD_TABLES,
    format_truncated_list,
    get_workspace_table,
    validate_workspace_exists,
)
from mlflow.store.model_registry.dbmodels.models import (
    SqlRegisteredModel,
    SqlRegisteredModelTag,
    SqlWebhook,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlEvaluationDataset,
    SqlExperiment,
    SqlExperimentTag,
    SqlJob,
    SqlMCPServer,
    SqlMCPServerTag,
)
from mlflow.store.workspace.sqlalchemy_store import _WORKSPACE_ROOT_MODELS


@dataclass(frozen=True)
class MoveResult:
    """Result of a move-resources operation."""

    names: list[str]
    row_count: int
    # Populated only when artifact_policy="copy": one plan per moved experiment.
    artifact_plans: tuple[ExperimentArtifactPlan, ...] = ()
    copied_file_count: int = 0


@dataclass(frozen=True)
class _ResourceSpec:
    """Metadata for a movable resource type."""

    model: type
    name_column: str = "name"
    tag_model: type | None = None
    # Column used to join the tag table back to the resource table.
    # For experiments the tag table joins via experiment_id, not workspace+name.
    tag_join_column: str | None = None
    child_tables: tuple[str | tuple[str, str], ...] = ()
    child_name_column: str = "name"
    has_unique_name: bool = True

    @property
    def table(self) -> sa.Table:
        return self.model.__table__

    @property
    def tag_table(self) -> sa.Table | None:
        return self.tag_model.__table__ if self.tag_model else None


# Per-model spec.  Keyed by ORM model class; the CLI resource type is derived
# from model.__tablename__ (e.g. "experiments", "registered_models").
# Models in _WORKSPACE_ROOT_MODELS without an entry here are silently skipped;
# a unit test verifies the omissions are intentional.
#
# Gateway resources (secrets, endpoints, model_definitions, budget_policies)
# are intentionally excluded because they have inter-table FK dependencies
# that make moving them independently unsafe.  They can be added later with
# proper dependency-aware handling.
_SPEC_BY_MODEL: dict[type, _ResourceSpec] = {
    SqlExperiment: _ResourceSpec(
        model=SqlExperiment,
        name_column=SqlExperiment.name.key,
        tag_model=SqlExperimentTag,
        tag_join_column=SqlExperimentTag.experiment_id.key,
    ),
    SqlRegisteredModel: _ResourceSpec(
        model=SqlRegisteredModel,
        name_column=SqlRegisteredModel.name.key,
        tag_model=SqlRegisteredModelTag,
        child_tables=tuple(MODEL_CHILD_TABLES),
    ),
    SqlEvaluationDataset: _ResourceSpec(
        model=SqlEvaluationDataset,
        name_column=SqlEvaluationDataset.name.key,
        has_unique_name=False,
    ),
    SqlWebhook: _ResourceSpec(
        model=SqlWebhook,
        name_column=SqlWebhook.name.key,
        has_unique_name=False,
    ),
    SqlJob: _ResourceSpec(
        model=SqlJob,
        name_column=SqlJob.job_name.key,
        has_unique_name=False,
    ),
    SqlMCPServer: _ResourceSpec(
        model=SqlMCPServer,
        name_column=SqlMCPServer.name.key,
        tag_model=SqlMCPServerTag,
        child_tables=(
            "mcp_server_versions",
            "mcp_server_tags",
            "mcp_server_version_tags",
            "mcp_server_aliases",
            ("mcp_access_endpoints", "server_name"),
        ),
    ),
}

_RESOURCE_SPECS: dict[str, _ResourceSpec] = {
    model.__tablename__: _SPEC_BY_MODEL[model]
    for model in _WORKSPACE_ROOT_MODELS
    if model in _SPEC_BY_MODEL
}
RESOURCE_TYPE_CHOICES = sorted(_RESOURCE_SPECS)


def _tag_names_subquery(
    spec: _ResourceSpec,
    source_workspace: str,
    tags: list[tuple[str, str]],
) -> sa.Select:
    """Build a SELECT subquery of resource names matching ALL given tags.

    The intersection logic uses ``GROUP BY … HAVING COUNT`` so the entire
    resolution stays in SQL and avoids materializing a large parameter list.
    """
    table = spec.table
    tag_table = spec.tag_table
    unique_tags = list(dict.fromkeys(tags))

    tag_conditions = sa.or_(*[
        sa.and_(tag_table.c.key == k, tag_table.c.value == v) for k, v in unique_tags
    ])

    if spec.tag_join_column:
        # Experiments: tags reference the resource via a surrogate key
        # (experiment_id), so we JOIN back to the resource table to get
        # the name and scope by the resource table's workspace column.
        id_col = spec.tag_join_column
        name_col = table.c[spec.name_column]
        subq = (
            sa
            .select(name_col)
            .select_from(table.join(tag_table, table.c[id_col] == tag_table.c[id_col]))
            .where(table.c.workspace == source_workspace)
            .where(tag_conditions)
            .group_by(name_col)
        )
    else:
        # Registered models: the tag table carries workspace + name
        # directly, so we can query it without joining the parent table.
        name_col = tag_table.c[spec.name_column]
        subq = (
            sa
            .select(name_col)
            .where(tag_table.c.workspace == source_workspace)
            .where(tag_conditions)
            .group_by(name_col)
        )

    if len(unique_tags) > 1:
        subq = subq.having(sa.func.count() == len(unique_tags))

    return subq


def _resolve_names(
    conn,
    spec: _ResourceSpec,
    workspace: str,
    names: list[str] | None = None,
) -> set[str]:
    """Return resource names in *workspace*, optionally filtered to *names*."""
    table = spec.table
    name_col = table.c[spec.name_column]
    stmt = sa.select(name_col).where(table.c.workspace == workspace)
    if names is not None:
        stmt = stmt.where(name_col.in_(names))
    return {row[0] for row in conn.execute(stmt).fetchall()}


def _find_conflicts(
    conn,
    spec: _ResourceSpec,
    source_workspace: str,
    target_workspace: str,
    name_filter: list[str] | sa.Select | None = None,
) -> list[str]:
    """Return source resource names that already exist in *target_workspace*.

    *name_filter* can be a ``list`` (literal names), a ``Select`` subquery,
    or ``None`` (move-all, falls back to a source-workspace subquery).
    SQLAlchemy's ``in_()`` handles both lists and subqueries transparently.
    """
    if not spec.has_unique_name:
        return []
    table = spec.table
    name_col = table.c[spec.name_column]
    stmt = sa.select(name_col).where(table.c.workspace == target_workspace)

    if name_filter is not None:
        stmt = stmt.where(name_col.in_(name_filter))
    else:
        source_subq = sa.select(name_col).where(table.c.workspace == source_workspace)
        stmt = stmt.where(name_col.in_(source_subq))

    return [row[0] for row in conn.execute(stmt.order_by(name_col)).fetchall()]


def _plan_move(
    conn,
    spec: _ResourceSpec,
    resource_type: str,
    source_workspace: str,
    target_workspace: str,
    names: list[str] | None,
    tags: list[tuple[str, str]] | None,
    verbose: bool,
):
    """Validate the move and resolve the matched names, name filter and row count."""
    validate_workspace_exists(conn, source_workspace)
    validate_workspace_exists(conn, target_workspace)

    # Fail fast with a clear message if the resource table lacks a
    # workspace column (DB not migrated to workspace-enabled schema).
    get_workspace_table(conn, spec.table.name)

    # Build a unified name filter: a SQL subquery (--tag), a small
    # literal list (--name), or None (move-all).  SQLAlchemy's in_()
    # handles lists and Select objects identically, so every subsequent
    # query uses the same one-branch pattern.
    if tags:
        name_filter = _tag_names_subquery(spec, source_workspace, tags)
        matched = {row[0] for row in conn.execute(name_filter).fetchall()}
    elif names:
        matched = _resolve_names(conn, spec, source_workspace, names)
        name_filter = list(matched)
    else:
        matched = _resolve_names(conn, spec, source_workspace)
        name_filter = None

    if not matched:
        return matched, name_filter, 0

    if conflicts := _find_conflicts(conn, spec, source_workspace, target_workspace, name_filter):
        formatted = format_truncated_list(
            [repr(name) for name in conflicts],
            max_rows=None if verbose else 10,
        )
        raise RuntimeError(
            f"Move aborted: the following {resource_type} already exist "
            f"in workspace {target_workspace!r} and would conflict: "
            f"{formatted}\n"
            "Rename or remove the conflicting resources in the target "
            "workspace, then retry."
        )

    table = spec.table
    name_col = table.c[spec.name_column]
    count_stmt = (
        sa.select(sa.func.count()).select_from(table).where(table.c.workspace == source_workspace)
    )
    if name_filter is not None:
        count_stmt = count_stmt.where(name_col.in_(name_filter))
    row_count = conn.execute(count_stmt).scalar()
    return matched, name_filter, row_count


def _execute_move(
    conn,
    spec: _ResourceSpec,
    source_workspace: str,
    target_workspace: str,
    name_filter,
) -> None:
    """Flip the workspace column on the resource table and its child tables."""
    table = spec.table
    name_col = table.c[spec.name_column]

    def _filtered(stmt, col, _nf=name_filter):
        return stmt.where(col.in_(_nf)) if _nf is not None else stmt

    conn.execute(
        _filtered(
            table
            .update()
            .where(table.c.workspace == source_workspace)
            .values(workspace=target_workspace),
            name_col,
        )
    )

    # Explicitly update child tables because not all backends honour
    # ON UPDATE CASCADE (e.g. SQLite without the foreign_keys pragma).
    # Each entry is either a table name str (uses spec.child_name_column)
    # or a (table_name, column_name) tuple for non-standard FK columns.
    for entry in spec.child_tables:
        if isinstance(entry, tuple):
            child_table_name = entry[0]
            col_name = entry[1]
        else:
            child_table_name = entry
            col_name = spec.child_name_column
        child = get_workspace_table(conn, child_table_name)
        conn.execute(
            _filtered(
                child
                .update()
                .where(child.c.workspace == source_workspace)
                .values(workspace=target_workspace),
                child.c[col_name],
            )
        )


def move_resources(
    engine: sa.Engine,
    source_workspace: str,
    target_workspace: str,
    resource_type: str,
    names: list[str] | None = None,
    tags: list[tuple[str, str]] | None = None,
    dry_run: bool = False,
    *,
    verbose: bool = False,
    artifact_policy: str = "preserve",
    default_artifact_root: str | None = None,
) -> MoveResult:
    """
    Move resources of *resource_type* from *source_workspace* to *target_workspace*.

    Filter by *names* or *tags* (mutually exclusive).  When neither is provided
    all resources of the type in the source workspace are moved.

    With ``artifact_policy="copy"`` (experiments only), the experiments' artifact
    objects are copied to the artifact root resolved for the target workspace and
    the stored artifact URIs (experiment, runs, logged models, trace tags) are
    rewritten to the new prefix. The copy happens before any database change and
    the old prefix is never deleted. ``default_artifact_root`` must match the
    tracking server's ``--default-artifact-root`` when the target workspace has no
    ``default_artifact_root`` of its own.

    Returns a :class:`MoveResult` with ``names`` (sorted list of distinct
    resource names that were moved or would be moved) and ``row_count`` (the
    number of rows in the root resource table that were moved; child-table
    rows such as model versions or tags are not included in this count).
    For resource types whose names are not unique, ``row_count`` may exceed
    ``len(names)`` when multiple rows share the same name.
    """
    if source_workspace == target_workspace:
        raise RuntimeError("Source and target workspaces must be different.")

    spec = _RESOURCE_SPECS.get(resource_type)
    if spec is None:
        raise RuntimeError(
            f"Unknown resource type {resource_type!r}. "
            f"Valid types: {', '.join(RESOURCE_TYPE_CHOICES)}"
        )

    if names and tags:
        raise RuntimeError("--name and --tag are mutually exclusive.")

    if tags and spec.tag_table is None:
        raise RuntimeError(f"Resource type {resource_type!r} does not support tag filtering.")

    if artifact_policy not in ("preserve", "copy"):
        raise RuntimeError(f"Unknown artifact policy {artifact_policy!r}.")

    if artifact_policy == "copy":
        if resource_type != "experiments":
            raise RuntimeError(
                "--artifact-policy copy is only supported for --resource-type experiments."
            )
        return _move_experiments_with_artifact_copy(
            engine,
            spec,
            source_workspace=source_workspace,
            target_workspace=target_workspace,
            names=names,
            tags=tags,
            dry_run=dry_run,
            verbose=verbose,
            default_artifact_root=default_artifact_root,
        )

    with engine.begin() as conn:
        matched, name_filter, row_count = _plan_move(
            conn, spec, resource_type, source_workspace, target_workspace, names, tags, verbose
        )
        if not matched:
            return MoveResult(names=[], row_count=0)
        if not dry_run:
            _execute_move(conn, spec, source_workspace, target_workspace, name_filter)

    return MoveResult(names=sorted(matched), row_count=row_count)


def _move_experiments_with_artifact_copy(
    engine: sa.Engine,
    spec: _ResourceSpec,
    *,
    source_workspace: str,
    target_workspace: str,
    names: list[str] | None,
    tags: list[tuple[str, str]] | None,
    dry_run: bool,
    verbose: bool,
    default_artifact_root: str | None,
) -> MoveResult:
    """Move experiments with artifact relocation, in three phases.

    Phase 1 plans the move and the per-experiment artifact relocation in a
    read-only transaction. Phase 2 copies and verifies artifact objects outside
    any transaction, since the copy can be long-running. Phase 3 re-validates the
    matched set and applies the workspace flip plus URI rewrites atomically. A
    failure in any phase leaves the source data intact, and rerunning after a
    partial copy reuses the already-copied objects.
    """
    with engine.connect() as conn:
        matched, _, row_count = _plan_move(
            conn, spec, "experiments", source_workspace, target_workspace, names, tags, verbose
        )
        if not matched:
            return MoveResult(names=[], row_count=0)
        plans = build_experiment_artifact_plans(
            conn, sorted(matched), source_workspace, target_workspace, default_artifact_root
        )

    if dry_run:
        return MoveResult(names=sorted(matched), row_count=row_count, artifact_plans=tuple(plans))

    copied_file_count = 0
    for plan in plans:
        copied_file_count += copy_experiment_artifacts(plan)

    with engine.begin() as conn:
        rematched, name_filter, row_count = _plan_move(
            conn,
            spec,
            "experiments",
            source_workspace,
            target_workspace,
            sorted(matched),
            None,
            verbose,
        )
        if rematched != matched:
            raise RuntimeError(
                "Experiments changed while artifacts were being copied "
                f"(expected {sorted(matched)}, found {sorted(rematched)}). "
                "No database changes were made. Copied artifacts remain at the "
                "target root and a rerun will reuse them."
            )
        # Matching names are not enough: an experiment could have been moved away
        # and its name recreated while the copy ran, in which case the name flip
        # and the id-keyed URI rewrites would target different experiments. Require
        # the exact (id, name, artifact_location) identities the plans were built from.
        experiments = spec.table
        current_identities = {
            row.experiment_id: (row.name, row.artifact_location)
            for row in conn.execute(
                sa.select(
                    experiments.c.experiment_id,
                    experiments.c.name,
                    experiments.c.artifact_location,
                ).where(
                    experiments.c.workspace == source_workspace,
                    experiments.c.name.in_(sorted(matched)),
                )
            )
        }
        planned_identities = {
            plan.experiment_id: (plan.experiment_name, plan.old_root) for plan in plans
        }
        if current_identities != planned_identities:
            raise RuntimeError(
                "Experiment identities or artifact locations changed while artifacts "
                "were being copied. No database changes were made. Copied artifacts "
                "remain at the target root and a rerun will reuse them."
            )
        _execute_move(conn, spec, source_workspace, target_workspace, name_filter)
        for plan in plans:
            rewrite_experiment_artifact_uris(conn, plan)

    return MoveResult(
        names=sorted(matched),
        row_count=row_count,
        artifact_plans=tuple(plans),
        copied_file_count=copied_file_count,
    )
