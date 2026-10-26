#!/usr/bin/env bash

# Run one task in its own process group and wait for its real terminal status.
# It is intentionally small so each entrypoint has identical INT/TERM semantics.

ACTIVE_TASK_PID=""

_stop_active_task() {
    local signal=$1
    local code=$2
    if [[ -n "${ACTIVE_TASK_PID}" ]]; then
        echo "Received ${signal}; stopping task process group ${ACTIVE_TASK_PID}." >&2
        kill -"${signal}" -- "-${ACTIVE_TASK_PID}" 2>/dev/null || true
        wait "${ACTIVE_TASK_PID}" 2>/dev/null || true
        ACTIVE_TASK_PID=""
    fi
    exit "${code}"
}

install_task_signal_traps() {
    trap '_stop_active_task INT 130' INT
    trap '_stop_active_task TERM 143' TERM
}

run_logged_foreground() {
    local log_file=$1
    shift
    if ! command -v setsid >/dev/null 2>&1; then
        echo "setsid is required to supervise foreground task process groups." >&2
        return 1
    fi
    setsid "$@" >>"${log_file}" 2>&1 &
    ACTIVE_TASK_PID=$!
    local status=0
    wait "${ACTIVE_TASK_PID}" || status=$?
    ACTIVE_TASK_PID=""
    return "${status}"
}
