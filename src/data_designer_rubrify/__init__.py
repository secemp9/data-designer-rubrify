from __future__ import annotations

from data_designer.plugins import Plugin, PluginType

plugin = Plugin(
    impl_qualified_name="data_designer_rubrify.generator.RubrifyCellGenerator",
    config_qualified_name="data_designer_rubrify.config.RubrifyColumnConfig",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
