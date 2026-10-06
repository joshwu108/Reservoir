"""Adapters that connect Reservoir buffers to external training frameworks.

Each submodule depends on its framework only when imported, so the core
package installs and imports without any of them. Available adapters:

- ``reservoir.integrations.trl``: replay for TRL's ``GRPOTrainer``.
- ``reservoir.integrations.verl``: replay for verl's ``RayPPOTrainer`` under GRPO.
"""
