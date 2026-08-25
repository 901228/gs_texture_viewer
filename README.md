# texture_viewer

## 編譯

### 需求

| 項目   | 版本                            |
| ------ | ------------------------------- |
| CMake  | >= 3.16                         |
| C++    | C++20（Windows 上以 MSVC 為主） |
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

---

## 執行與操作

### 3D 視圖的操作

| 操作             | 動作               |
| ---------------- | ------------------ |
| 滾輪             | 縮放               |
| 中鍵拖曳         | 旋轉               |
| Shift + 中鍵拖曳 | 平移               |
| 左鍵點擊 / 拖曳  | 用筆刷**選取**面   |
| 右鍵點擊 / 拖曳  | 用筆刷**取消選取** |

`camera` 分頁裡有 `Focus on Model`（把視點對回模型中心）與 `Move Speed`（對數尺度）。

### parameterization 面板（貼圖預覽）的操作

| 操作         | 動作                               |
| ------------ | ---------------------------------- |
| 滾輪         | 縮放貼圖                           |
| 按住 `R`     | 進入旋轉模式，拉出角度線後點擊確認 |
| 左鍵拖曳     | 移動貼圖                           |
| 拖曳中按右鍵 | 取消這次移動                       |

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
- 貼圖的新增是手動編輯 `textures.toml`
- 貼圖清單：點縮圖選取，再點一次取消選取

選到一組 PBR 貼圖後，會多出這組貼圖自己的參數：

| 參數                  | 說明                                                        |
| --------------------- | ----------------------------------------------------------- |
| `roughness scale`     | 乘在取樣到的 roughness 上（0–2）                            |
| `Height Mode`         | `None` / `Parallax Occlusion` / `Tessellation Displacement` |
| `height scale`        | 高度強度（`Height Mode` 非 `None` 時才出現）                |
| `tessellation level`  | 細分等級 1–64（只在 `Tessellation Displacement` 出現）      |
| `invert displacement` | 反轉位移方向（模型法線朝內時用）                            |

---

## textures.toml

貼圖清單存在 **repo 根目錄的 `textures.toml`**：

```shell
cp textures.sample.toml textures.toml   # Windows: copy textures.sample.toml textures.toml
```

`textures.sample.toml` 是範本，裡面有註解與兩個範例 entry，把 `path` 改成你自己的
貼圖資料夾即可。

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

| 欄位          | 型別   | 說明                                                                               |
| ------------- | ------ | ---------------------------------------------------------------------------------- |
| `path`        | string | 貼圖**資料夾**路徑。實務上請用絕對路徑（相對路徑會相對於執行時的工作目錄，不可靠） |
| `basecolor`   | string | 相對於 `path` 的檔名，RGB 或 RGBA                                                  |
| `normal`      | string | 相對於 `path` 的檔名，以 RGB 讀入                                                  |
| `height`      | string | 相對於 `path` 的檔名，以單通道（R）讀入                                            |
| `roughness`   | string | 相對於 `path` 的檔名，以單通道（R）讀入                                            |
| `mask`        | string | 相對於 `path` 的檔名，以單通道（R）讀入，決定貼花範圍                              |
| `heightScale` | float  | 可省略，省略時預設 `0.01`                                                          |

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
