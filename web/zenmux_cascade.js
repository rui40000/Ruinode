// ZenMux 节点「厂商 → 模型」二级级联
// ==================================
// 后端把全量模型标签（含价格）都放进 model 下拉以通过校验；
// 本扩展在前端按 vendor 收窄 model 的候选列表。
// 约定：模型标签第一个空格前是 "vendor/model" 形式的 id，
//       故 label.split("/")[0] 即厂商名，与 vendor 下拉的取值同源。
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Ruinode.ZenMuxCascade",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "ZenMuxAPINode") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const vendorW = this.widgets?.find((w) => w.name === "vendor");
            const modelW = this.widgets?.find((w) => w.name === "model");
            if (!vendorW || !modelW) return r;

            // 全量标签快照（此时 options.values 是后端给的完整列表）
            const allLabels = (modelW.options.values || []).slice();
            const vendorOf = (label) => String(label).split("/")[0];

            // resetIfMissing：当前值不在收窄后的列表时是否重置为列表首项。
            // 手动切厂商 → 重置；加载旧工作流 → 保留（旧价格标签后端能解析）。
            const applyFilter = (vendor, resetIfMissing) => {
                const filtered = allLabels.filter((l) => vendorOf(l) === vendor);
                modelW.options.values = filtered.length ? filtered : allLabels.slice();
                if (resetIfMissing && !modelW.options.values.includes(modelW.value)) {
                    modelW.value = modelW.options.values[0];
                    modelW.callback?.(modelW.value);
                }
                this.setDirtyCanvas?.(true, true);
            };

            const origCallback = vendorW.callback;
            vendorW.callback = function (value) {
                const r2 = origCallback?.apply(this, arguments);
                applyFilter(value, true);
                return r2;
            };

            // 新建节点：按默认厂商先收窄一次
            applyFilter(vendorW.value, false);

            // 加载已保存的工作流：configure 恢复 widget 值后再按真实厂商收窄，
            // 但不动用户保存的 model 选择
            const onConfigure = this.onConfigure;
            this.onConfigure = function () {
                const r3 = onConfigure?.apply(this, arguments);
                applyFilter(vendorW.value, false);
                return r3;
            };

            return r;
        };
    },
});
