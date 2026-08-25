# texture_viewer

一個用來研究**表面貼圖 / PBR** 與 **UV 參數化**（exponential map、harmonics）的桌面應用程式（C++20）。
以 OpenGL 算繪三角網格，UI 使用 Dear ImGui。

> 本分支為 `mesh_only`：原本的 Gaussian Splatting (3DGS)、CUDA rasterizer 與 CUDA 相依都已移除，
> 只保留 mesh + texture 的功能。

---

## 目錄

- [編譯](#編譯)
- [執行與操作](#執行與操作)
- [textures.toml](#texturestoml)
- [PBR 貼圖要怎麼放](#pbr-貼圖要怎麼放)
- [專案結構](#專案結構)

---

## 編譯

### 需求

| 項目 | 版本 |
| --- | --- |
| CMake | >= 3.16 |
| C++ | C++20（Windows 上以 MSVC 為主） |
| OpenGL | 4.x（使用 tessellation shader） |

所有第三方函式庫都是 git submodule，第一次 clone 後務必初始化：

```shell
git submodule update --init --recursive
```

### 方法一：Visual Studio generator（Windows 最穩定）

CMake 會自己找到 MSVC，不需要 Developer Command Prompt：

```shell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug
# 產物：build/src/Debug/texture_viewer.exe（同時會複製一份到 build/Debug/）
```

### 方法二：script/ 內的便利腳本（Ninja）

需要編譯器已在 PATH 上（先跑過 `vcvarsall.bat` 或用 Developer Command Prompt）：

```shell
script/build.bat Debug        # configure + build，Release 亦可
script/configure.bat Debug    # 只做 configure
script/clean.bat              # 清除 build/
```

POSIX 環境有對應的 `.sh` 版本。

### 方法三：VSCode

1. 安裝擴充套件 [`CMake Tools`](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)
2. 執行 `CMake: Configure`
3. 執行 `CMake: Debug`

### 執行

執行檔不需要任何命令列參數，直接啟動就會開啟主視窗；mesh 與貼圖都是在 UI 裡載入。

資源路徑是靠編譯期定義的 `PROJECT_DIR`（見 `CMakeLists.txt`）解析的，也就是 **`assets/`、`shaders/`、
`textures.toml` 一律是從 repo 根目錄讀取**，不受執行檔所在位置影響。副作用是：改完 shader 不用重新編譯，
但把執行檔搬到別的機器上就找不到資源了。

> 沒有測試也沒有 lint target；程式碼風格由 `.clang-format` 控制（`ColumnLimit: 110`）。

---

## 執行與操作

### 視窗配置

視窗固定 1100×800（不可縮放），分成三塊：

```
+---------------------------------+------------------+
|                                 | parameterization |
|          main view              |  （貼圖預覽）      |
|      （3D 視圖 / 分頁）           +------------------+
|                                 |    controls      |
|                                 |  （側邊欄設定）    |
+---------------------------------+------------------+
```

`main view` 上方的分頁對應到不同的 panel：

| 分頁 | 說明 |
| --- | --- |
| **GLB View** | 載入 `assets/models/` 底下遞迴掃到的 `.glb` / `.gltf`，含多光源系統 |
| **Model View** | 載入 `.obj`（預設 `assets/models/armadillo.obj`） |

### 3D 視圖的操作

| 操作 | 動作 |
| --- | --- |
| 滾輪 | 縮放 |
| 中鍵拖曳 | 旋轉 |
| Shift + 中鍵拖曳 | 平移 |
| 左鍵點擊 / 拖曳 | 用筆刷**選取**面 |
| 右鍵點擊 / 拖曳 | 用筆刷**取消選取** |

`camera` 分頁裡有 `Focus on Model`（把視點對回模型中心）與 `Move Speed`（對數尺度）。

### parameterization 面板（貼圖預覽）的操作

| 操作 | 動作 |
| --- | --- |
| 滾輪 | 縮放貼圖 |
| 按住 `R` | 進入旋轉模式，拉出角度線後點擊確認 |
| 左鍵拖曳 | 移動貼圖 |
| 拖曳中按右鍵 | 取消這次移動 |

沒有選取任何貼圖時，游標會顯示為禁止符號。

### controls 側邊欄

**render**
- `Model`：切換要載入的模型（GLB View）
- `wire`：疊加線框
- `render selected only`：只畫被選取的面
- `flip normals`：翻轉法線（貼花也會跟著水平鏡射）
- `decal normal only`：貼花的 tangent frame 建在幾何法線上，忽略 glb 本身的 normal map
- `Rendering Mode`：`Mesh` / `TextureCoords` / `Texture`

**light**（GLB View 為多光源；Model View 只有一個 2D gizmo 決定光線方向 + `Light Intensity`）
- 預設是三點打光（key / fill / rim），可 `Add Light` / `delete`，最多 8 盞（`MAX_LIGHTS`）
- 每盞燈有 `enabled`、`animate`、`azimuth`、`elevation`、`intensity`、顏色
- `animate orbit` + `orbit speed`：讓有勾 `animate` 的燈繞著模型轉
- `show lights in view`：在畫面上疊出光源 gizmo

**camera**：`Focus on Model`、`Move Speed`

**textures**
- `Select Mode`：
  - `Point` — 每次點擊會先清空選取，適合 exponential map（以點擊點為圓心展開）
  - `Faces` — 累積式筆刷選取，適合 harmonics
- `Brush Size`：筆刷半徑（1–60）
- `Clear Selection`：清空選取（只在 `Faces` 模式可用）
- `Auto Solve Texture Coords`：選取變動後自動重算（只在 `Faces` 模式可用）
- `Method`：`Harmonics` / `ExpMap`
- `Calculate Parameterization`：手動觸發一次求解
- `Add Texture`：開啟資料夾選擇器加入一組 PBR 貼圖（見下一節）
- 貼圖清單：點縮圖選取，再點一次取消選取

選到一組 PBR 貼圖後，會多出這組貼圖自己的參數：

| 參數 | 說明 |
| --- | --- |
| `roughness scale` | 乘在取樣到的 roughness 上（0–2） |
| `Height Mode` | `None` / `Parallax Occlusion` / `Tessellation Displacement` |
| `height scale` | 高度強度（`Height Mode` 非 `None` 時才出現） |
| `tessellation level` | 細分等級 1–64（只在 `Tessellation Displacement` 出現） |
| `invert displacement` | 反轉位移方向（模型法線朝內時用） |

---

## textures.toml

貼圖清單存在 **repo 根目錄的 `textures.toml`**（路徑寫死在
`src/utils/texture/texture_editor.hpp` 的 `PROJECT_DIR "/textures.toml"`）。
檔案不存在也沒關係，第一次按 `Add Texture` 時會自動建立。

> ⚠️ `textures.toml` 在 `.gitignore` 裡，屬於個人的本機清單，不會進版控。

### 格式

每組 PBR 貼圖是 `pbrList` 陣列裡的一個 table：

```toml
[[pbrList]]
path = 'D:\research\texture_viewer\assets\texture\my_logo'
basecolor = 'basecolor.png'
normal = 'normal.png'
height = 'height.png'
roughness = 'roughness.png'
mask = 'logo_mask.png'
heightScale = 0.1
```

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `path` | string | 貼圖**資料夾**路徑。實務上請用絕對路徑（相對路徑會相對於執行時的工作目錄，不可靠） |
| `basecolor` | string | 相對於 `path` 的檔名，RGB 或 RGBA |
| `normal` | string | 相對於 `path` 的檔名，以 RGB 讀入 |
| `height` | string | 相對於 `path` 的檔名，以單通道（R）讀入 |
| `roughness` | string | 相對於 `path` 的檔名，以單通道（R）讀入 |
| `mask` | string | 相對於 `path` 的檔名，以單通道（R）讀入，決定貼花範圍 |
| `heightScale` | float | 可省略，省略時預設 `0.01` |

注意事項：

- 這六個檔名欄位請**全部填齊**。載入時只有 `basecolor` / `normal` / `height` 缺少會安全跳過該筆，
  缺 `roughness` 或 `mask` 會直接踩到空指標。
- 檔名只會取 basename，寫成子路徑（`sub/normal.png`）沒有用，實際仍會去 `path` 底下找同名檔。
- 通道數在 Debug build 會用 `assert` 檢查：normal 必須是 3 通道，height / roughness / mask 必須是 1 通道。
  灰階圖請確實輸出成單通道 PNG/JPG。
- 支援的副檔名：`.png`、`.jpg`、`.jpeg`。
- 檔案是由程式**整份覆寫**的（`PBRTexture::saveTextureList`）：每次 `Add Texture` 都會把整個
  `pbrList` 重寫一遍，字串一律用單引號。手改是可以的，但要在程式沒開的時候改，
  否則會被下一次存檔蓋掉。目前 UI 沒有刪除貼圖的按鈕，**要刪貼圖就是手動編輯這個檔案**。
- 檔案裡還可能有一個 `texturesList = [...]`（純字串陣列，非 PBR 的單張貼圖模式）。
  目前兩個 panel 都是以 PBR 模式建立 `TextureEditor`，所以這個 key 不會被用到，但存檔時會被保留。

---

## PBR 貼圖要怎麼放

### 一組貼圖 = 一個資料夾

一組 PBR 貼圖就是一個資料夾，裡面放五張圖。資料夾名稱會成為 UI 上顯示的名稱。
資料夾可以放在任何地方（`textures.toml` 記的是絕對路徑），但建議統一放在 `assets/texture/<名稱>/`：

```
assets/texture/my_logo/
├── basecolor.png     # RGB / RGBA，貼花的顏色
├── normal.png        # RGB，tangent space normal map
├── height.png        # 單通道灰階，給 parallax / tessellation displacement 用
├── roughness.png     # 單通道灰階
└── logo_mask.png     # 單通道灰階，白 = 有貼花、黑 = 露出底下的材質
```

> `assets/` 整個目錄在 `.gitignore` 裡，貼圖與模型都不會進版控。

### 用 `Add Texture` 加入

按 `Add Texture` 會跳出**資料夾**選擇器（不是選檔案）。程式會在你選的資料夾裡依序找這些檔名
（副檔名 `.png` / `.jpg` / `.jpeg` 都試）：

| 用途 | 期待的檔名 |
| --- | --- |
| basecolor | `basecolor.*` |
| normal | `normal.*` |
| height | `height.*` |
| roughness | `roughness.*` |
| mask | `logo_mask.*` |

`basecolor` / `normal` / `height` / `logo_mask` 任一缺少就會拒絕加入，並在 log 裡印出找不到的路徑。
用這個方式加入時 `heightScale` 會被設成 `0`，之後在 UI 上調整、或直接改 `textures.toml` 都可以。

> 檔名不一樣（例如既有資料夾裡的 `result.png` / `mask.jpg`）也不是不行，
> 但那就得**手動在 `textures.toml` 裡加一筆**，因為資料夾選擇器只認上表的名字。

### 其他

- `metalness.*` 目前**不會**被讀取，放了也沒有效果。
- 五張圖的尺寸不需要一致（各自獨立取樣），但通常做成同解析度比較好對齊。
- `mask` 是決定貼花邊界的關鍵：`basecolor` 的 alpha 不會被當成遮罩用，透明區域請確實反映在 `mask` 上。

---

## 專案結構

```
src/
├── main_window.*            # GLFW 視窗 + ImGui context + panel 清單，Run() 是算繪迴圈
├── panel/
│   └── page_panel/
│       ├── gltf_panel.*     # GLB View：.glb/.gltf + 多光源
│       └── model_panel.*    # Model View：.obj
└── utils/
    ├── camera/              # trackball / imguizmo 相機
    ├── gl/                  # program、framebuffer、screen quad 等薄封裝
    ├── imgui/               # sidebar、image selectable、tool line 等自製元件
    ├── mesh/                # OpenMesh 網格、hit test、solve_uv / expmap / harmonics
    └── texture/             # ImageTexture / PBRTexture / TextureEditor
shaders/                     # GLSL（gltf/ 底下是 GLB View 用的 vert/tesc/tese/frag）
assets/                      # 模型、貼圖、字型、圖示（未進版控）
extern/                      # 第三方 submodule
script/                      # build / configure / clean 腳本
```

慣例：

- 資源路徑一律用 `Utils::Path::getAssetsPath` / `getShaderPath`（`src/utils/utils.hpp`），不要寫死路徑。
- log 用 `src/utils/logger.hpp` 的 `INFO` / `DEBUG` / `WARN` / `ERROR`，不要用 `printf` / `std::cout`。
- enum 透過 `magic_enum` 產生 ImGui combo（`Utils::enumToImGuiCombo`）。
