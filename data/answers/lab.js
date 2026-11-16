// SPDX-License-Identifier: GPL-3.0-or-later · Copyright (C) 2026 Пантелеев Г.Д.
/* Практические работы 1–6 курса С. А. Тарасова (github.com/jzcurious/hsys-tasks):
   что сделано в репозитории hsys, ключевой код, результаты и что спрашивают на защите. */
function LT(rows){ return tbl(["Что спросят","Что ответить"], rows); }
Object.assign(A, {
"lab-1": `
<p><b>Цель</b>: базовые навыки CUDA — одномерная сетка нитей, динамическая память устройства; паттерн <b>data + view</b>.</p>
${F("sv_dataview")}
${H("Классы")}
${UL(["<b>Data&lt;AtomT&gt;</b> — владелец массива в глобальной памяти: " + K("cudaMalloc") + " в конструкторе, " + K("cudaFree") + " в деструкторе (RAII), копирование (новый буфер + cudaMemcpyDeviceToDevice) и перемещение (забираем указатель, у источника — nullptr), copy_to_host / copy_from_host",
  "<b>VectorView&lt;AtomT&gt;</b> — указатель и размер, методы " + K("__host__ __device__") + "; тривиально копируемый — поэтому его можно передать в ядро по значению",
  "<b>Vector&lt;AtomT&gt;</b> — фасад: " + K("std::shared_ptr<Data>") + " (агрегация — несколько векторов могут делить данные) и собственный VectorView (композиция)"])}
${H("Ядро и запуск")}
${CODE(`template <typename AtomT>
__global__ void add_kernel(VectorView<AtomT> a, VectorView<AtomT> b, VectorView<AtomT> c) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < a.size()) c[i] = a[i] + b[i];      // лишние нити последнего блока ничего не делают
}
int threads = 256;
int blocks = (length + threads - 1) / threads;  // округление вверх
add_kernel<<<blocks, threads>>>(a.view(), b.view(), c.view());
cudaDeviceSynchronize();`)}
${H("Тесты и замеры")}
${UL(["Google Test: n ∈ {1, 2, 3, 127, 128, 129, 512, 1024, 1029} — границы блока 128 и 1024 и «некратные» размеры; эталон Eigen::VectorXf, isApprox с точностью 10<sup>−6</sup>",
  "Google Benchmark в репозитории: n = 2<sup>10</sup> … 2<sup>20</sup> (" + K("RangeMultiplier(2)->Range(1 << 10, 1 << 20)") + "); в BM_AddGPU внутри цикла копирования на GPU, ядро и копирование обратно"])}
${F2("sv_w1_time","Время: GPU медленнее при всех n","sv_w1_speed","Ускорение от 0,005 до 0,64")}
${H("Как объяснить результаты")}
${UL(["На каждый элемент — 1 сложение и 12 байт (два чтения и запись float): задача упирается в память, а с копированиями — в PCIe (≈ 16 ГБ/с против сотен ГБ/с у видеопамяти)",
  "При малых n время GPU почти постоянно (≈ 15 мкс): это запуск ядра и вызовы cudaMemcpy, а не счёт",
  "Скачок CPU при n = 2<sup>20</sup>: три вектора по 4 МиБ = 12 МиБ — ровно L3 процессора, данные перестают помещаться в кэш",
  "Теория: сложность O(n) у обоих; GPU выиграл бы, если бы данные уже лежали на устройстве (задание требует исключать копирования из замера)"])}
${CARD("lab1")}
${H("На защите")}
${LT([["Зачем View, если есть Data?","ядру нужен лёгкий объект «указатель + размер», который копируется в параметры запуска; Data владеет памятью и не должен копироваться побитово (двойной cudaFree)"],
  ["Почему проверка i &lt; n?","нитей blocks · 256 ≥ n; лишние нити последнего блока вышли бы за массив"],
  ["Сколько блоков при n = 1029?","(1029 + 255) / 256 = 5 блоков, 1280 нитей, 251 простаивают"],
  ["Что будет без cudaDeviceSynchronize?","запуск асинхронный: время и результат прочитаются до окончания ядра (cudaMemcpy, правда, сам ждёт)"],
  ["Почему GPU медленнее?","копирования по PCIe и накладные расходы запуска больше, чем выигрыш на одной операции сложения"]])}
`,

"lab-2": `
<p><b>Цель</b>: двумерные сетки нитей; класс <b>Matrix</b> (фасад над Data и MatrixView) и наивное ядро умножения матриц без разделяемой памяти.</p>
${H("MatrixView и Matrix")}
${UL(["Хранение <b>row-major</b>: элемент (i, j) лежит по индексу " + K("i * ncols + j") + "; у Eigen по умолчанию column-major — эталон надо приводить",
  "MatrixView: указатель, nrows, ncols, " + K("operator()(i, j)") + " для хоста и устройства; " + K("static_assert(std::is_trivially_copyable_v<MatrixView<float>>)"),
  "Matrix: shared_ptr&lt;Data&gt; + MatrixView; Data теперь проверяет ошибки через CUDA_CHECK и безопасна при исключениях (copy-and-swap)"])}
${H("Наивное ядро")}
${CODE(`template <typename AtomT>
__global__ void kernel_matmul_naive(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) {
    const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;   // y — строки
    const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;   // x — столбцы
    if (row >= c.nrows() || col >= c.ncols()) return;
    AtomT sum{0};
    for (std::size_t p = 0; p < a.ncols(); ++p) sum += a(row, p) * b(p, col);
    c(row, col) = sum;
}
const dim3 block(16, 16);                                  // 256 нитей
const dim3 grid((c.ncols() + 15) / 16, (c.nrows() + 15) / 16);`)}
${UL(["Каждая нить — один элемент C: k умножений и k сложений, итого 2·m·n·k операций — O(n³) для квадратных",
  "x-индекс нити — столбец: соседние нити варпа читают соседние b(p, col) и пишут соседние c(row, col) — объединённый доступ; a(row, p) у всего варпа одинаковый — широковещательное чтение"])}
${F("sv_coalesce")}
${H("Тесты и бенчмарки")}
${UL(["Тесты: m, n, k ∈ {1, 2, 3, 127, 128, 129, 512} — декартово произведение через " + K("::testing::Combine") + ", 7³ = 343 сочетания; эталон Eigen::MatrixXf, isApprox 10<sup>−5</sup>",
  "Бенчмарки: n × n, n ∈ {16 … 1024}; время GPU — CUDA Events (класс CudaTimer), " + K("state.SetIterationTime") + " и " + K("UseManualTime()") + "; выделение и копирования — вне замера"])}
${F2("sv_w2_time","Время умножения: рост ∝ n³","sv_w2_speed","Ускорение относительно Eigen")}
${H("Как объяснить результаты")}
${UL(["При n = 16, 32 GPU медленнее: работы на ~1 блок, время ≈ 3 мкс — накладные расходы запуска",
  "С n = 64 GPU обгоняет Eigen; при n = 256 и 512 ускорение 12,3 раза, при 1024 — 11,6",
  "При удвоении n время растёт примерно в 8 раз (n³): 586 мкс → 5043 мкс для 512 → 1024 (в 8,6 раза)",
  "GPU при n = 1024: 2·1024³ / 5,04 мс ≈ 426 Гфлопс; Eigen ≈ 37 Гфлопс",
  "Наивное ядро ограничено памятью: каждый элемент A и B читается из глобальной памяти n раз — это исправляет работа 3"])}
${CARD("lab2")}
${H("На защите")}
${LT([["Сколько блоков при m = n = 129?","⌈129/16⌉ = 9 по x и по y → 81 блок"],["Почему x — столбцы?","для объединения обращений к B и C: соседние нити варпа идут по строке"],["Что такое Combine?","генератор Google Test: декартово произведение наборов значений → параметризованные тесты TEST_P"],["Зачем CUDA Events?","ядро запускается асинхронно; события ставятся в поток и показывают время на устройстве"],["Сложность алгоритма?","O(m·n·k), для квадратных — O(n³)"]])}
`,

"lab-3": `
<p><b>Цель</b>: разделяемая память и синхронизация нитей блока — <b>блочное (tiled) умножение матриц</b> ${K("kernel_matmul_shmem")}. Выбор алгоритма — через паттерн <b>стратегия</b>.</p>
${F("sv_tiling")}
${H("Ядро")}
${CODE(`template <class AtomT, unsigned TileSize>   // TileSize = 16
__global__ void kernel_matmul_shmem(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) {
  __shared__ AtomT a_tile[TileSize][TileSize];
  __shared__ AtomT b_tile[TileSize][TileSize];
  const unsigned tx = threadIdx.x, ty = threadIdx.y;
  const std::size_t row = blockIdx.y * TileSize + ty, col = blockIdx.x * TileSize + tx;
  AtomT sum{0};
  for (std::size_t tile = 0; tile < a.ncols(); tile += TileSize) {
    a_tile[ty][tx] = (row < a.nrows() && tile + tx < a.ncols()) ? a(row, tile + tx) : AtomT{0};
    b_tile[ty][tx] = (tile + ty < b.nrows() && col < b.ncols()) ? b(tile + ty, col) : AtomT{0};
    __syncthreads();                       // плитки загружены всеми нитями
    for (unsigned p = 0; p < TileSize; ++p) sum += a_tile[ty][p] * b_tile[p][tx];
    __syncthreads();                       // плитки прочитаны — можно перезаписывать
  }
  if (row < c.nrows() && col < c.ncols()) c(row, col) = sum;
}`)}
${H("Почему быстрее")}
${UL(["Блок 16 × 16 загружает плитку A и плитку B один раз, а каждый элемент плитки используется 16 нитями — обращений к глобальной памяти в 16 раз меньше",
  "Арифметическая интенсивность растёт: на шаге 2 КиБ загрузки дают 16·16·16·2 = 8192 операции ≈ 4 флопа на байт",
  "Shared memory — на SM, задержка как у L1; " + K("a_tile[ty][p]") + " — одинаковый адрес у нитей с одним ty (широковещание), " + K("b_tile[p][tx]") + " — соседние банки, конфликтов нет"])}
${H("Синхронизация")}
${UL(["Первый __syncthreads — нельзя считать, пока соседи не загрузили свои элементы",
  "Второй — нельзя загружать следующую плитку, пока соседи читают текущую (гонка «запись после чтения»)",
  "__syncthreads должны вызывать все нити блока: поэтому выход за границы матрицы обрабатывается записью нулей, а не " + K("return") + " до барьера"])}
${H("Стратегия")}
${CODE(`template <class StrategyT, class AtomT>
concept MatMulStrategy = requires(const MatrixView<AtomT>& a, const MatrixView<AtomT>& b, MatrixView<AtomT>& c) {
  { StrategyT::multiply(a, b, c) } -> std::same_as<void>;
};
struct NaiveMatMul { ... };   struct ShmemMatMul { static constexpr unsigned tile_size = 16; ... };`)}
${UL(["Тесты: те же размеры, что в работе 2 (1, 2, 3, 127, 128, 129, 512), для каждой стратегии; isApprox 10<sup>−5</sup>",
  "Бенчмарки: n = 16 … 1024, график сложности для новой и наивной версии и ускорения shmem относительно naive"])}
${CARD("lab3")}
${H("На защите")}
${LT([["Сколько shared на блок?","2 · 16 · 16 · 4 байта = 2048 байт = 2 КиБ (float)"],["Что будет без второго __syncthreads?","быстрые нити перезапишут плитку, пока медленные её читают — неверный результат"],["Можно ли поставить return для нитей за границей?","нет: они не дойдут до __syncthreads — неопределённое поведение (зависание)"],["Сколько шагов цикла при k = 129?","⌈129/16⌉ = 9 плиток"],["Почему не Cooperative Groups?","задание требует __syncthreads — базовый барьер блока"]])}
`,

"lab-4": `
<p><b>Цель</b>: программирование <b>тензорных ядер</b> через CUDA WMMA (Warp Matrix Multiply-Accumulate) — ядро ${K("kernel_matmul_wmma")}.</p>
${F("sv_wmma")}
${H("Как устроено")}
${UL(["Тензорное ядро за такт вычисляет D = A·B + C для маленьких матриц; WMMA даёт доступ на уровне <b>варпа</b>: фрагменты 16 × 16 × 16",
  "A и B — " + K("half") + " (FP16), аккумулятор — " + K("float") + " (точность суммы не теряется)",
  "Все функции " + K("*_sync") + " — коллективные: их вызывают все 32 нити варпа, данные фрагмента распределены по регистрам нитей",
  "Блок нитей = 4 × 2 варпа (" + K("dim3 block(4 * 32, 2)") + " = 256 нитей) → плитка C 32 строки × 64 столбца, каждый варп — свой фрагмент 16 × 16"])}
${CODE(`wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
wmma::fill_fragment(acc_frag, 0.0f);
for (std::size_t p = 0; p < a.ncols(); p += 16) {
  wmma::load_matrix_sync(a_frag, &a(tile_row, p), lda);
  wmma::load_matrix_sync(b_frag, &b(p, tile_col), ldb);
  wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
}
wmma::store_matrix_sync(stage, acc_frag, 16, wmma::mem_row_major);   // float в shared
__syncwarp();
for (unsigned i = lane; i < 256; i += 32)
  c(tile_row + i / 16, tile_col + i % 16) = __float2half(stage[i]);`)}
${H("Детали реализации")}
${UL(["Аккумулятор float нельзя сразу записать в half-матрицу C — поэтому " + K("__shared__ __align__(32) float staging[8][256]") + " (8 варпов × 256 значений = 8 КиБ) и преобразование " + K("__float2half"),
  "Требования WMMA: размеры кратны 16 (иначе " + K("std::invalid_argument") + "), указатель выровнен на 32 байта, ldm кратен 8 для half",
  "Условие выхода " + K("if (tile_row >= c.nrows() || tile_col >= c.ncols()) return;") + " одинаково для всех нитей варпа — варп выходит целиком, коллективные вызовы не ломаются",
  "Стратегия по умолчанию: " + K("default_matmul_strategy<half>") + " → WmmaMatMul, для остальных типов — ShmemMatMul (специализация шаблона)"])}
${H("Тесты и точность")}
${UL(["m, n, k ∈ {16, 32, 64, 128, 256, 512}; эталон — Eigen с типом " + K("Eigen::half") + " в RowMajor, точность 10<sup>−2</sup>",
  "Почему такая точность: у half 11 значащих бит (≈ 3 десятичных знака), входы округляются до half, результат тоже хранится в half; накопление во float ошибку не увеличивает",
  "Измеренная относительная ошибка — около 3·10<sup>−3</sup> при k = 512"])}
${CARD("lab4")}
${H("На защите")}
${LT([["Сколько блоков для 512 × 512?","по x: 512 / 64 = 8, по y: 512 / 32 = 16 → 128 блоков"],["Зачем аккумулятор float?","сумма 512 произведений в half теряла бы точность и могла переполниться (максимум half ≈ 65 504)"],["Что такое ldm?","leading dimension — шаг между строками в элементах (для row-major — число столбцов)"],["Почему WMMA только с CC 7.0?","тензорные ядра появились в Volta; T4 (7.5) их имеет"],["Чем WMMA лучше shmem?","тензорное ядро делает 16·16·16 FMA за несколько тактов — у T4 65 Тфлопс FP16 против 8,1 Тфлопс FP32"]])}
`,

"lab-5": `
<p><b>Цель</b>: разделяемая память и барьеры, программирование уровня <b>варпа</b> и <b>широковещание регистров</b> (${K("__shfl_down_sync")}) на задаче <b>редукции</b> — суммы элементов вектора.</p>
${F("sv_reduce")}
${H("Общая схема (обе версии)")}
${UL(["<b>Два прохода</b>: первый — блоки по 256 нитей пишут частичные суммы в partial[blocks]; второй — один блок сворачивает partial в result[0]",
  "Число блоков " + K("min(⌈n / 256⌉, 1024)") + " — не больше 1024, поэтому второму проходу хватает одного блока",
  "<b>Grid-stride loop</b>: нить суммирует элементы i, i + gridDim.x·256, … — любой n обрабатывается фиксированной сеткой (до 8·2<sup>28</sup> элементов)"])}
${H("(1) kernel_vecred_nobr — дерево в shared")}
${CODE(`__shared__ AtomT partial[BlockSize];                 // 256 значений
partial[tid] = thread_sum<AtomT, BlockSize>(in);
__syncthreads();
for (unsigned stride = BlockSize / 2; stride > 0; stride /= 2) {   // 128, 64, ..., 1
  if (tid < stride) partial[tid] += partial[tid + stride];
  __syncthreads();
}
if (tid == 0) out[blockIdx.x] = partial[0];`)}
<p>8 шагов (log₂256), 9 барьеров. Последовательная адресация (tid и tid + stride) — без конфликтов банков; активные нити — непрерывный диапазон, поэтому на последних шагах простаивают целые варпы, а не половины.</p>
${H("(2) kernel_vecred_br — обмен регистрами")}
${CODE(`template <class AtomT> __device__ AtomT warp_sum(AtomT value) {
  for (unsigned offset = 16; offset > 0; offset /= 2)          // 16, 8, 4, 2, 1
    value += __shfl_down_sync(0xffffffff, value, offset);      // взять регистр нити lane + offset
  return value;
}
const AtomT sum = warp_sum(thread_sum<AtomT, BlockSize>(in));
if (lane == 0) warp_sums[warp] = sum;                          // 8 сумм варпов
__syncthreads();                                               // единственный барьер
if (warp == 0) {
  const AtomT block_sum = warp_sum(lane < kWarps ? warp_sums[lane] : AtomT{0});
  if (lane == 0) out[blockIdx.x] = block_sum;
}`)}
${UL(["Внутри варпа нити обмениваются регистрами напрямую — без shared и без барьеров (маска 0xffffffff: участвуют все 32 нити)",
  "В shared — только 8 чисел, барьер — 1 вместо 9; меньше инструкций и простаивающих нитей",
  "Обе стратегии подчиняются концепту " + K("VecRedStrategy") + ": " + K("reduce(in, partial, result)") + " и " + K("partial_size(n)")])}
${H("Тесты и бенчмарки")}
${UL(["n ∈ {1, 2, 3, 127, 129, 512, 541, 1037}, " + K("EXPECT_NEAR") + " с точностью 10<sup>−4</sup>, эталон — sum() у Eigen; дополнительно n = 300 007 из единиц (несколько итераций grid-stride)",
  "Бенчмарки n = 8·2<sup>0</sup> … 8·2<sup>28</sup>: самый большой вектор — 2<sup>31</sup> float = 8 ГиБ; заполнение — ядром fill на устройстве",
  "Редукция ограничена памятью: на элемент 4 байта и 1 сложение; при больших n обе версии близки к пропускной способности видеопамяти, разница — при малых и средних n"])}
${CARD("lab5")}
${H("На защите")}
${LT([["Почему результат float отличается от Eigen?","порядок сложения другой — округления копятся иначе; поэтому EXPECT_NEAR, а не равенство"],["Что делает __shfl_down_sync(mask, v, 16)?","нить lane получает значение v нити lane + 16; нити 16–31 получают своё же (вне диапазона)"],["Зачем второй проход?","блоки не могут синхронизироваться между собой внутри ядра; сумма блоков — отдельным запуском"],["Сколько блоков при n = 1037?","⌈1037 / 256⌉ = 5"],["Почему BlockSize — степень двойки?","дерево делит число активных нитей пополам на каждом шаге"]])}
`,

"lab-6": `
<p><b>Цель</b>: CUDA-ядро, совместимое с <b>PyTorch</b>: пакетная многоканальная двумерная свёртка (кросс-корреляция) ${K("hsys::nn::functional::kernels::conv2d")} через LibTorch.</p>
${FX("Y[n, c<sub>out</sub>, h, w] = Σ<sub>c<sub>in</sub></sub> Σ<sub>r</sub> Σ<sub>q</sub> X̂[n, c<sub>in</sub>, h·s + r·d − p, w·s + q·d − p] · W[c<sub>out</sub>, c<sub>in</sub>, r, q]")}
<p>X̂ — вход с нулями за границами (отступ p); s — шаг (stride), d — расширение (dilation), k — размер фильтра.</p>
${CARD("lab6")}
${F("sv_conv")}
${H("Ядро")}
${UL(["Параметры k, s, p, d, block_size — <b>шаблонные</b>: известны при компиляции, циклы по фильтру разворачиваются, размеры shared — константы",
  "Сетка: " + K("blockIdx.x, blockIdx.y") + " — плитка выхода 16 × 16, " + K("blockIdx.z = n * C_out + c_out") + "; каждая нить — один выходной пиксель",
  "Для каждого c_in блок загружает в shared плитку входа " + K("tile = (bs - 1) * s + (k - 1) * d + 1") + " и фильтр k × k, затем " + K("__syncthreads()") + ", сумма, снова " + K("__syncthreads()"),
  "Нить читает " + K("input_tile[ty * s + r * d][tx * s + q * d]") + " — только shared, без глобальной памяти"])}
${H("Обёртка для PyTorch")}
${CODE(`TORCH_CHECK(input.is_cuda() && weights.is_cuda(), "conv2d: tensors must be on a CUDA device");
TORCH_CHECK(weights.size(1) == input.size(1), "conv2d: number of input channels mismatch");
const auto x = input.contiguous();                       // плотная раскладка NCHW
auto output = at::empty({batch, c_out, h_out, w_out}, x.options());
const c10::cuda::CUDAGuard guard(x.device());            // нужная GPU
const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hsys_conv2d", [&] {
  kernels::conv2d<scalar_t, k, s, p, d, 16><<<grid, block, 0, stream>>>(
      output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), ...);
});
C10_CUDA_KERNEL_LAUNCH_CHECK();`)}
${UL(["<b>PackedTensorAccessor32</b> — «view» тензора для ядра: указатель, размеры и шаги, индексация " + K("x[n][c][h][w]") + "; RestrictPtrTraits — указатель __restrict__",
  "<b>AT_DISPATCH_FLOATING_TYPES</b> — инстанцирует лямбду для float и double, " + K("scalar_t") + " — текущий тип",
  "Поток PyTorch, а не нулевой — ядро упорядочено с остальными операциями PyTorch"])}
${H("Тесты")}
${UL(["6 наборов (k, s, p, d): (3,2,1,1), (3,1,1,1), (7,2,3,1), (1,2,0,1), (3,1,2,2), (3,2,2,2) × 9 размеров (C<sub>in</sub>, H, W, C<sub>out</sub>) из ResNet-подобной сети = 54 сценария",
  "Эталон — " + K("torch::nn::functional::conv2d") + ", сравнение " + K("torch::allclose") + " с atol = rtol = 10<sup>−5</sup> (для half — 10<sup>−2</sup>); у cuDNN выключается TF32, иначе эталон сам неточен"])}
${H("На защите")}
${LT([["Размер выхода при H = 244, k = 7, s = 2, p = 3?","(244 + 6 − 6 − 1) / 2 + 1 = 122"],["Сторона плитки при k = 7, s = 2, d = 1?","15 · 2 + 6 · 1 + 1 = 37 — 37 · 37 · 4 = 5476 байт для float"],["Почему blockIdx.z = n · C_out + c_out?","у сетки три измерения: два заняты плиткой выхода, третье кодирует изображение и выходной канал; предел 65 535"],["Зачем contiguous()?","ядро считает раскладку плотной; транспонированный тензор иначе читался бы неверно"],["Как сделан padding?","при загрузке плитки: координата вне [0, H) × [0, W) — пишем 0"]])}
`
});
