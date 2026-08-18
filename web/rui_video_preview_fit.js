// Ruinode —— 修正节点内视频预览被裁切的问题
// ==========================================
// 现象：「加载透明视频」节点下方的视频预览按原始像素尺寸渲染，
//       容器 overflow:hidden，于是画面被裁掉一大块，只能看到局部。
//
// 根因：ComfyUI 官方给节点内视频预览的样式本来是对的——
//         .comfy-img-preview video { object-fit:contain; width:100%; height:100% }
//       但 comfyui-art-venture 插件（web/upload.js）为它自己的
//       LoadVideoFromUrl 节点注入了一条**同名全局选择器**的规则：
//         .comfy-img-preview video {
//           width:  var(--comfy-img-preview-width);
//           height: var(--comfy-img-preview-height);
//         }
//       这两个 CSS 变量只在 art-venture 自家节点的 DOM 上定义，
//       其它节点（本节点、原生 LoadVideo 等）上是空值。变量为空时
//       width/height 声明整条失效，<video> 便退回固有尺寸（如 768×768），
//       超出容器的部分被裁掉。
//
// 修法：注入两条规则。
//   1) 专属规则：只作用于本节点的预览容器，用 !important 保证必定生效。
//   2) 兜底规则：给那两个变量补上 100% 的回退值。变量有值时行为完全不变
//      （art-venture 自己的节点不受影响），只在变量为空这种「本来就坏掉」
//      的情况下才回到官方行为，因此对其它插件是纯修复、无副作用。

import { app } from "../../scripts/app.js";

const CONTAINER_CLASS = "rui-video-fit";
const STYLE_ID = "rui-video-preview-fit";

// 需要修正的节点类型；将来若有别的节点带视频预览，加进来即可
const TARGET_NODES = new Set(["RuiLoadVideoAlpha"]);

function injectStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
/* 1) 本节点专属：等比缩放到容器内完整显示 */
.comfy-img-preview.${CONTAINER_CLASS} video {
  width: 100% !important;
  height: 100% !important;
  object-fit: contain !important;
  object-position: center !important;
}
/* 容器自身也居中，视频比例与容器不一致时留白均匀 */
.comfy-img-preview.${CONTAINER_CLASS} {
  place-content: center !important;
  align-items: center !important;
}

/* 2) 全局兜底：给 art-venture 依赖的变量补回退值。
      变量有值时行为不变，为空时恢复成 ComfyUI 官方的 100%。*/
.comfy-img-preview video {
  width: var(--comfy-img-preview-width, 100%);
  height: var(--comfy-img-preview-height, 100%);
}
`;
  document.head.appendChild(style);
}

app.registerExtension({
  name: "Ruinode.VideoPreviewFit",

  init() {
    injectStyle();
  },

  async nodeCreated(node) {
    const type = node?.comfyClass || node?.constructor?.comfyClass || node?.type;
    if (!TARGET_NODES.has(type)) return;

    // 预览容器由 ComfyUI 的 useNodeVideo 在视频加载完成后才创建并赋值到
    // node.videoContainer。这里拦截该属性的写入，容器一出现就打上标记，
    // 不必轮询，也不受创建时机影响。
    let held = node.videoContainer;
    const mark = (el) => {
      if (el && el.classList && !el.classList.contains(CONTAINER_CLASS)) {
        el.classList.add(CONTAINER_CLASS);
      }
    };
    mark(held);

    Object.defineProperty(node, "videoContainer", {
      configurable: true,
      enumerable: true,
      get() {
        return held;
      },
      set(value) {
        held = value;
        mark(value);
      },
    });
  },
});
