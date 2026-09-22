"""Execution gate for a stored plan.

The client reports which services have completed and which have failed;
the gate then evaluates every remaining (pending) service against the
stored waves: a service may start exactly when all of its dependencies
have completed. A failed dependency blocks its dependents, and the
blockage propagates recursively to every transitive downstream service.
Each blocked service is reported with one localizable reason per blocking
dependency. The stored plan is only read, never modified.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import PlanningError
from .schemas import Issue

# Derived states for pending services (reported states are "completed"/"failed").
READY = "ready"  # all dependencies completed: may start now
BLOCKED = "blocked"  # at least one dependency not completed


@dataclass
class BlockedEntry:
    service: str
    wave: int  # 1-based wave in the stored plan
    reasons: list[Issue]


@dataclass
class GateOutput:
    earliest_executable_wave: int | None  # 1-based, None when nothing can start
    executable_services: list[str]  # sorted ids
    blocked: list[BlockedEntry]  # ordered by wave, then service id


def _validate_progress(
    services: list[str], completed: list[str], failed: list[str]
) -> None:
    """Reject unknown references and completed/failed conflicts."""
    issues: list[Issue] = []
    known = set(services)
    for field_name, reported in (("completed", completed), ("failed", failed)):
        for service in reported:
            if service not in known:
                issues.append(
                    Issue(
                        code="unknown_service",
                        message=(
                            f"Service '{service}' reported as {field_name} is "
                            "not part of the plan."
                        ),
                        constraint=field_name,
                        service=service,
                    )
                )
    for service in sorted(set(completed) & set(failed)):
        issues.append(
            Issue(
                code="conflicting_status",
                message=(
                    f"Service '{service}' is reported as both completed and "
                    "failed; keep it in exactly one of the lists."
                ),
                constraint="completed",
                service=service,
                context={"in_completed": True, "in_failed": True},
            )
        )
    if issues:
        raise PlanningError(
            "The gate request references unknown services or reports "
            "conflicting statuses.",
            issues,
        )


def evaluate_gate(
    services: list[str],
    edges: list[tuple[str, str]],
    waves: list[list[str]],
    completed: list[str],
    failed: list[str],
) -> GateOutput:
    """Decide which pending services may start and which are blocked.

    ``waves`` are the stored waves in order and ``edges`` the stored
    ``service -> depends_on`` pairs. Dependencies always sit in strictly
    earlier waves, so walking the waves in order resolves every dependency
    before its dependents. Reported states are taken as given; only pending
    services are judged.
    """
    _validate_progress(services, completed, failed)

    completed_set = set(completed)
    failed_set = set(failed)
    wave_of = {s: i + 1 for i, wave in enumerate(waves) for s in wave}
    deps_of: dict[str, list[str]] = {s: [] for s in services}
    for service, depends_on in edges:
        deps_of[service].append(depends_on)

    status: dict[str, str] = {}
    failed_roots: dict[str, set[str]] = {}  # blocked service -> root failures
    reasons_of: dict[str, list[Issue]] = {}

    for index, wave in enumerate(waves, start=1):
        for service in wave:
            if service in completed_set:
                status[service] = "completed"
                continue
            if service in failed_set:
                status[service] = "failed"
                continue

            reasons: list[Issue] = []
            roots: set[str] = set()
            for dep in sorted(deps_of[service]):
                dep_status = status[dep]
                if dep_status == "failed":
                    roots.add(dep)
                    reasons.append(
                        Issue(
                            code="dependency_failed",
                            message=(
                                f"Service '{service}' cannot start: dependency "
                                f"'{dep}' failed in wave {wave_of[dep]}."
                            ),
                            constraint="dependencies",
                            service=service,
                            related_services=[dep],
                            context={
                                "wave": index,
                                "dependency": dep,
                                "dependency_wave": wave_of[dep],
                                "dependency_status": dep_status,
                            },
                        )
                    )
                elif dep_status == BLOCKED and failed_roots[dep]:
                    # Recursive propagation of a failure downstream.
                    roots |= failed_roots[dep]
                    reasons.append(
                        Issue(
                            code="upstream_failed",
                            message=(
                                f"Service '{service}' cannot start: dependency "
                                f"'{dep}' is blocked by failed service(s) "
                                f"{', '.join(sorted(failed_roots[dep]))}."
                            ),
                            constraint="dependencies",
                            service=service,
                            related_services=[dep],
                            context={
                                "wave": index,
                                "dependency": dep,
                                "dependency_wave": wave_of[dep],
                                "dependency_status": dep_status,
                                "root_failed": sorted(failed_roots[dep]),
                            },
                        )
                    )
                elif dep_status != "completed":
                    reasons.append(
                        Issue(
                            code="dependency_incomplete",
                            message=(
                                f"Service '{service}' cannot start: dependency "
                                f"'{dep}' has not completed."
                            ),
                            constraint="dependencies",
                            service=service,
                            related_services=[dep],
                            context={
                                "wave": index,
                                "dependency": dep,
                                "dependency_wave": wave_of[dep],
                                "dependency_status": dep_status,
                            },
                        )
                    )

            if reasons:
                status[service] = BLOCKED
                failed_roots[service] = roots
                reasons_of[service] = reasons
            else:
                status[service] = READY

    executable = sorted(s for s in services if status[s] == READY)
    earliest = min((wave_of[s] for s in executable), default=None)
    blocked = [
        BlockedEntry(service=s, wave=wave_of[s], reasons=reasons_of[s])
        for s in sorted(reasons_of, key=lambda name: (wave_of[name], name))
    ]
    return GateOutput(
        earliest_executable_wave=earliest,
        executable_services=executable,
        blocked=blocked,
    )
