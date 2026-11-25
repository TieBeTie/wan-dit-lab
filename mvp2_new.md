<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

# Исследование · MVP2

Переписано 12.11.2026 по стилю research-blackboard-style из [Исследование MVP2.canvas](Исследование%20MVP2.canvas). Оригинал не менялся. Сверено 16.11.2026 с отчётом (§2, §11–14, §18–31), STATE и конфигами; что не найдено в источниках, помечено «в отчёте нет». Обозначения общие с [MVP1](mvp1_new.md), новые вводятся ниже.

## Контекст и трудность

$$
\begin{aligned}
\mathrm{cfg}_{\rm MVP1}&\in\mathcal C_{\rm mem},&&\text{Рассмотрим работающую систему MVP1,}\\
\mathcal L(\mathrm{cfg})&=\mathrm{LPIPS}_{\rm align}+(1-\mathrm{DINO}),&&\text{у которой качество к эталону не измерялось.}\\
704\times1280&\mapsto H\times W,&&\text{Ускоряем сжатием кадра обученной модели,}\\
\mathcal L\uparrow\ \text{при}\ h\times w\downarrow,&&&\text{и трудность в том, что потеря растёт со сжатием,}\\
\mathcal L_{\rm floor}&=0.26\ /\ 0.18\ \text{на повторе},&&\text{а повтор того же конфига сначала давал другой ролик.}
\end{aligned}
$$

## Цель MVP2

Найти конфигурацию с наименьшей потерей к эталону $704\times1280$ при $\mathrm{FPS}\ge15$ на RTX 4060 Laptop.

## Задача

$$
\begin{aligned}
\min_{\mathrm{cfg}\in\mathcal B_b}\quad&\overline{\mathcal L}_1(\mathrm{cfg}),\qquad b\in\{\mathrm{PI},\mathrm{NTK}\}\\
\text{при условиях}\quad&\mathrm{FPS}(\mathrm{cfg})\ge15,\quad M_{\rm VRAM}\le8\ \text{ГБ},\quad M_{\rm RAM}\le16\ \text{ГБ},\\
\text{старт}\quad&\mathrm{cfg}_{\rm start}=\mathrm{point\mbox{-}b2},\qquad\text{шаг меняет один фактор.}
\end{aligned}
$$

Задача 2.1: шкала потери (пол и потолок). Задача 2.2: стартовая точка. Задача 2.3: спуск по разрешению и профиль RoPE. Задача 2.4: удешевление памяти видео. Задача 2.5: скорость без изменения результата.

## Обозначения MVP2

Рассмотрим сравнение ролика конфигурации с роликом эталона того же seed.

$$
\underbrace{\mathrm{cfg}_0}_{\text{эталон }704\times1280},\quad
\underbrace{s}_{\text{seed}},\quad
\underbrace{\mathcal L(a,b)}_{\text{потеря ролика }a\text{ к ролику }b},\quad
\underbrace{\mathcal L_1}_{\text{потеря первого чанка}},\quad
\underbrace{\overline{\mathcal L}_1}_{\text{среднее по 3 seed и 4 сценам}}
$$
$$
\underbrace{M_a=\tfrac{h}{h'}}_{\text{множитель оси }a\in\{H,W\}},\qquad
\underbrace{\Sigma_{\rm atomic},\ \Sigma_{\rm ordered}}_{\text{сложение частичных сумм: порядок не задан, задан}},\qquad
\underbrace{t_1,\ t_2=833,\ t=0}_{\text{время шагов денойзинга и прохода записи}}
$$

Теперь рассмотрим, чем меряем потерю, а затем от какой точки спускаемся.

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Измерение потери · MVP2

Насколько потеря различает конфигурации? Рассмотрим пол (повтор) и потолок (другой seed).

$$
\underbrace{\mathcal L_{\rm floor}=\mathcal L\bigl(\mathrm{video}'(\mathrm{cfg}_0,s),\mathrm{video}(\mathrm{cfg}_0,s)\bigr)}_{\text{повтор той же конфигурации}},\qquad
\underbrace{\mathcal L_{\rm ceil}=\mathcal L\bigl(\mathrm{video}(\mathrm{cfg}_0,s'),\mathrm{video}(\mathrm{cfg}_0,s)\bigr)}_{\text{другой seed}},\qquad
\underbrace{\tfrac{\mathcal L-\mathcal L_{\rm floor}}{\mathcal L_{\rm ceil}-\mathcal L_{\rm floor}}}_{\text{доля зазора}}
$$

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| Не применяем: $\mathcal L_{\rm floor}=0.26$<br>$\displaystyle \mathrm{cfg}_0\mapsto\mathrm{cfg}_0$, $\Sigma_{\rm atomic}$ | $\displaystyle \begin{aligned}&\text{Рассмотрим сложение float из плиток ядра.}\\&\underbrace{\overline K}_{\text{наше ядро подготовки ключей}},\quad\underbrace{xW_{\rm down}}_{\text{ветка низкого ранга Nunchaku}}\\&\text{порядок слагаемых не задан}\end{aligned}$ | Проверяем, даёт ли один конфиг один ролик. | $\displaystyle \begin{aligned}&\underbrace{36\text{–}38\%}_{\text{расхождение латента, чанк 1}},\ \text{с блока 0}\\&\mathcal L_{\rm floor}=\underbrace{0.26}_{\text{лиса}},\ \underbrace{0.18}_{\text{чашка}}\end{aligned}$<br>[Отчёт, §3](docs/report-2026-11-11-loss-704-608-for-fable.md) |
| <span class="wan-kept">Применяем</span> в экспериментах<br>$\displaystyle \Sigma_{\rm atomic}\mapsto\Sigma_{\rm ordered}$<br>`kernels.lora_sum` | $\displaystyle \begin{aligned}&\text{Было: плитки складывают в общий адрес атомарно.}\\&\text{Стало: каждая пишет }p_k\text{, затем }((p_1+p_2)+\cdots)+p_m.\end{aligned}$ | Наше: правка ядра Nunchaku и своего ядра. В Live остаётся $\Sigma_{\rm atomic}$. | $\displaystyle \begin{aligned}&\mathcal L_{\rm floor}=0,\ \text{латент и MP4 побитно равны}\\&\underbrace{T_{\rm block}:38.0\to38.8\ \mathrm{ms},\ \mathrm{FPS}:-2\%}_{608\times1120}\end{aligned}$<br>[Отчёт, §11](docs/report-2026-11-11-loss-704-608-for-fable.md) |
| <span class="wan-kept">Применяем</span> как шкалу<br>$\displaystyle \mathrm{video}_0\mapsto\operatorname{resize}_{608}(\mathrm{video}_0)$ | $\displaystyle \begin{aligned}&\text{Уменьшаем готовый эталон, без генерации.}\\&\mathcal L\bigl(\operatorname{resize}_{608}(\mathrm{video}_0),\mathrm{video}_0\bigr)\end{aligned}$ | Цена самой потери пикселей, нижняя граница для ступени 608. | $\displaystyle \underbrace{0.013\ /\ 0.008}_{\text{лиса / чашка, 3 чанка, seed 0}}$ |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathcal L\mapsto\overline{\mathcal L}_1$ | $\displaystyle \begin{aligned}&s\in\{0,1,2\},\quad\text{чашка, лиса, космонавт, две лисы}\\&\text{цепочки в порядке эталона: шум seed идёт подряд}\end{aligned}$ | Потеря набирается уже в первом чанке, точка стоит около 5 с. | $\displaystyle \begin{aligned}&\underbrace{\overline{\mathcal L}_1=0.360,\ \overline{\mathcal L}=0.398}_{608\times1120}\\&\mathcal L_{\rm ceil}=\underbrace{0.736}_{\text{чанк 1}},\ \underbrace{0.753}_{\text{ролик}}\end{aligned}$<br>[Отчёт, §27](docs/report-2026-11-11-loss-704-608-for-fable.md) |
| <span class="wan-kept">Применяем</span> в экспериментах<br>$\displaystyle \epsilon\mapsto\operatorname{spectral}_{h\times w}(\epsilon_{22\times40})$<br>`descent.noise` | $\displaystyle \begin{aligned}&\text{Было: свежий шум на сетке }h\times w.\\&\text{Стало: шум эталона с обрезанным спектром.}\end{aligned}$ | У эталона и сжатого кадра совпадают низкие частоты шума. В Live шум свежий. | $\displaystyle \begin{aligned}&\underbrace{0.43\ \mathrm{s}}_{\text{CPU на чанк}},\quad\underbrace{\mathrm{FPS}:8.60\to9.73}_{608\times1120,\ \text{свежий шум}}\\&\text{FPS меряем со свежим, потерю со спектральным}\end{aligned}$ |
| <span class="wan-kept">Применяем</span> вывод<br>$\displaystyle M:1\mapsto1+10^{-5}$ | $\displaystyle \begin{aligned}&\text{Одно малое возмущение, три варианта.}\\&\underbrace{\delta=\lVert\mathrm{lat}'-\mathrm{lat}\rVert/\lVert\mathrm{lat}\rVert}_{\text{чанк 1, }704\times1280,\ \text{лиса},\ \Sigma_{\rm ordered}}\end{aligned}$ | Кто усиливает малую ошибку: двухшаговая модель или 4 бита. Вывод: сравнивать только при $\Sigma_{\rm ordered}$. | $\displaystyle \delta=\underbrace{0.35}_{\mathrm{W4A4+Sage}},\ \underbrace{0.18}_{\mathrm{FP8+Sage}},\ \underbrace{0.16}_{\mathrm{FP8+SDPA}}$<br>Усиливает модель, W4A4 удваивает. |

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Стартовая точка · MVP2

От какой конфигурации спускаться? Рассмотрим конфигурацию как набор факторов при $704\times1280$, KV FP8, T5 BF16.

$$
\underbrace{\mathrm{cfg}=(W_{\rm DiT},W_{\rm T5},\mathrm{KV},\mathrm{Attn},\mathrm{RoPE},H\times W)}_{\text{факторы}},\qquad
\underbrace{\mathrm{cfg}^{(k)}\mapsto\mathrm{cfg}^{(k+1)}}_{\text{один фактор за шаг}}
$$

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{Attn}_{\rm self}:\mathrm{SDPA}\mapsto\mathrm{Sage\,2}$ | $\displaystyle \begin{aligned}&\text{Рассмотрим вершину A′: FP8, SDPA.}\\&\text{Меняем только ядро self-attention.}\end{aligned}$ | Самый дешёвый по качеству шаг. | $\displaystyle \underbrace{\mathrm{FPS}:1.30\to2.17,\quad24\text{–}27\ \mathrm{dB}}_{\text{PSNR чанка 1 к A′}}$<br>[E29](notes/research/cards/E29.%20Вершина%20A′%20и%20спуск%20B,%20FP8,%20704×1280.md) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle W_{\rm DiT}:\mathrm{FP8}\mapsto\mathrm{W4A4}$ | $\displaystyle \underbrace{\mathrm{LongLive\,2.0\ S2}}_{\text{дистиллирован в NVFP4}}$ | Для этого чекпойнта ближе к FP32 не значит лучше видео. | $\displaystyle \mathrm{FPS}:2.17\to5.41$<br>В просмотре сетки не хуже FP8.<br>[E30](notes/research/cards/E30.%20Стартовая%20точка%20W4A4%20704×1280%20и%20отложенные%20шаги.md) |
| Отложено: выбор пользователя<br>$\displaystyle W_{\rm T5}:\mathrm{BF16}\mapsto\mathrm{INT8}$ | $\displaystyle \underbrace{T_{\rm prompt}}_{\text{время смены промпта, вне цикла чанков}}$ | Быстрее смена промпта, FPS почти не меняется. | $\displaystyle \mathrm{FPS}:5.41\to5.58,\quad T_{\rm prompt}:2.86\to1.63\ \mathrm{s}$<br>В просмотре промпт соответствует хуже.<br>[E30](notes/research/cards/E30.%20Стартовая%20точка%20W4A4%20704×1280%20и%20отложенные%20шаги.md) |
| Не применяем: $\mathrm{FPS}:5.41\to5.45$<br>$\displaystyle \mathrm{RoPE}\to\mathrm{Attn}_{\rm Sage}\mapsto\mathrm{RoPE}\overset{\rm fuse}{\hookrightarrow}\mathrm{Attn}_{\rm Sage}$ | $\displaystyle \begin{aligned}&\text{Было в MVP1: слитое ядро.}\\&\underbrace{0.029\%}_{\text{разница со стартом на вызов}}\end{aligned}$ | Скорости не даёт, а видео расходится как другой seed. | [E30](notes/research/cards/E30.%20Стартовая%20точка%20W4A4%20704×1280%20и%20отложенные%20шаги.md) |

$$
\underbrace{\mathrm{cfg}_{\rm start}=\mathrm{point\mbox{-}b2}}_{\text{W4A4, Sage 2 self, T5 BF16, KV FP8, }704\times1280},\qquad\mathrm{FPS}=5.41.
$$

[Конфиг стартовой точки](configs/experiments/director-original/point-b2-w4a4-sage-704x1280.json)

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Смена разрешения и RoPE · MVP2

Как уменьшить сетку токенов с наименьшей потерей? Рассмотрим частоты RoPE одной пространственной оси.

$$
\underbrace{(h,w):(22,40)\mapsto(h',w')}_{\text{сетка токенов}},\qquad
\underbrace{\theta_j=10000^{-2j/42},\ j=0,\ldots,20}_{\text{частоты оси}},\qquad
\theta'_{a,j}=\begin{cases}M_a\theta_j,&\mathrm{PI},\\M_a^{\,j/20}\theta_j,&\mathrm{NTK},\\M_a^{\,\mu+(1-\mu)j/20}\theta_j,&\mathrm{mix}(\mu).\end{cases}
$$

Временная ось не меняется. Обзор статей в [MVP1](mvp1_new.md).

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| Не применяем: $M^{\star}$ не растёт<br>$\displaystyle \theta_j\mapsto M^{j/20}\theta_j$, NTK<br>[YaRN, App. A.2](https://arxiv.org/html/2309.00071) | $\displaystyle \begin{aligned}&\text{Высокие частоты оставляем, низкие растягиваем.}\\&M^{\star}(h')=\arg\min_M\mathcal L\end{aligned}$ | Гипотеза: лучшее $M$ растёт со сжатием. | $\displaystyle \begin{aligned}&\underbrace{M^{\star}=1.2\text{–}1.3}_{672,\ 640,\ 608}\\&\underbrace{\overline{\mathcal L}_1=0.436}_{\text{отложенные сцены, }M=1.2}\end{aligned}$<br>[Спуск 09.11](D:/Experiments/DIT/outputs/res-descent-2026-11-09/REPORT.md) |
| Не применяем: устойчивого улучшения нет<br>$\displaystyle \mathrm{shift}\mapsto\sqrt{\tfrac{H'W'}{HW}}\,\mathrm{shift}$<br>[Scaling Rectified Flow, (23)](https://arxiv.org/html/2403.03206#S5.SS3.SSS2) | $\displaystyle \underbrace{5\mapsto4.11\mapsto3.63}_{576\times1056}$ | Меньше токенов, меньше shift. | Оставляем $\mathrm{shift}=5$. |
| Не применяем: подгонка под обучающие сцены<br>$\displaystyle (M_H,M_W,\mu)\mapsto\arg\min\overline{\mathcal L}_1$, CMA-ES | $\displaystyle \begin{aligned}&\underbrace{\text{чашка, лиса}}_{\text{обучающие}},\ \underbrace{\text{космонавт, две лисы}}_{\text{отложенные}}\\&\underbrace{144\ \text{точки}}_{\text{seed 0, 1}},\ \underbrace{\text{seed 2}}_{\text{отложенный}}\end{aligned}$ | Есть ли профиль лучше одного числа на ось. | $\displaystyle \begin{aligned}&(1.134,\ 1.104,\ 1):\ \underbrace{0.338}_{\text{обучающие}},\ \underbrace{0.382}_{\text{отложенные}}\\&\text{плато с 3-го поколения}\end{aligned}$<br>[Отчёт, §14](docs/report-2026-11-11-loss-704-608-for-fable.md) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle M_H=\tfrac{22}{h'},\ M_W=\tfrac{40}{w'}$<br>PI по отношению сеток<br>[Position Interpolation, §2.3](https://arxiv.org/html/2306.15595) | $\displaystyle \begin{aligned}&\text{Позиции сжатого кадра растягиваем до обученной сетки.}\\&\underbrace{M_a(h'-1)\approx h-1}_{\text{тот же диапазон позиций}}\end{aligned}$ | Без подбора: множитель задан сетками. Объект сохраняет положение и масштаб. | $\displaystyle \begin{aligned}&\underbrace{0.387}_{\text{PI, отложенные, }608}\text{ против }\underbrace{0.382}_{\rm CMA}\\&\underbrace{0.442,\ 0.619,\ 0.607}_{\text{PI, без интерполяции, }M=1.1;\ 512,\ \text{seed 0}}\end{aligned}$<br>В просмотре объекты стоят там же, где в эталоне.<br>Оценка MVP1 «лучше $M=1.05$–$1.1$» дана глазами на одной сцене; здесь $M=1.1$ хуже по потере ($0.607$ против $0.442$). |
| Не применяем: знак разницы зависит от ядра cross-attention, FPS $-6\%$<br>$\displaystyle 17\times30\mapsto17\times31$ | $\displaystyle \begin{aligned}&\underbrace{M_H=1.294,\ M_W=1.333}_{544\times960}\ \mapsto\ \underbrace{1.294,\ 1.290}_{544\times992}\end{aligned}$ | Гипотеза: разное растяжение осей портит потерю. | $\displaystyle \begin{aligned}&\underbrace{0.426\ \text{против}\ 0.414}_{\text{cross SDPA: }544\times960,\ 544\times992}\\&\underbrace{0.426\ \text{против}\ 0.436}_{\text{cross Sage 2: }544\times960,\ 544\times992}\end{aligned}$<br>Гипотеза не подтверждена и не отвергнута: разница $0.010$–$0.012$ меняет знак.<br>[Отчёт, §27, §28, §30](docs/report-2026-11-11-loss-704-608-for-fable.md) |

Лестница разрешений: PI по отношению сеток, $\Sigma_{\rm ordered}$, Sage 2 только в self-attention, FPS последнего чанка лисы со свежим шумом.

| $H\times W$ | $h\times w$ | FPS, $F_{\rm win}=32$ | FPS, $F_{\rm win}=24$ | $\overline{\mathcal L}_1$ | $\overline{\mathcal L}_1/\mathcal L_{\rm ceil}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $704\times1280$ | $22\times40$ | $5.71$ | $6.58$ | $0$ | $0$ |
| $608\times1120$ | $19\times35$ | $8.51$ | $9.61$ | $0.360$ | $0.49$ |
| $576\times1056$ | $18\times33$ | $9.90$ | $11.14$ | $0.393$ | $0.53$ |
| $544\times992$ | $17\times31$ | $11.83$ | $13.20$ | $0.414$ | $0.56$ |
| $544\times960$ | $17\times30$ | $12.53$ | $13.94$ | $0.426$ | $0.58$ |
| $512\times928$ | $16\times29$ | $14.18$ | $15.73$ | $0.483$ | $0.66$ |

Шаг $544\mapsto512$ стоит $0.057$ потери, предыдущие шаги $0.012$–$0.033$. Лестница снята с cross SDPA. С Sage 2 в cross-attention: $512\times928$ даёт $0.484$, $544\times960$ даёт $0.426$, $544\times992$ даёт $0.436$.

Потеря первого чанка не зависит от окна: первые три чанка при $F_{\rm win}=24$ и $32$ побитно равны. Окна различает только дрейф на 12 чанках.

[Сетки](D:/Experiments/DIT/outputs/speed-2026-11-12/grids/) · [Гипотеза NTK](notes/research/NTK-zoom-hypothesis.md)

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Память видео · MVP2

Можно ли удешевить память видео, не теряя сцену? Рассмотрим три прохода одного чанка.

$$
(\epsilon_{n+1},\mathrm{ctx},\mathrm{KVvid}_n)\xrightarrow{t_1}\widehat{\mathrm{lat}}_{n+1}\xrightarrow{t_2=833}\mathrm{lat}_{n+1}\xrightarrow{t=0,\ \text{запись}}\mathrm{KVvid}_{n+1},\qquad
\underbrace{T_{\rm chunk}=3\cdot30\cdot T_{\rm block}+T_{\rm dec}}_{\text{три прохода по 30 блокам и декодер}}
$$

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| Не применяем: падение на смене сцены, дрейф 3 из 12 на других ступенях<br>$\displaystyle F_{\rm win}:32\mapsto24$ | $\displaystyle \begin{aligned}&\text{Кольцо короче на чанк:}\\&\underbrace{F_{\rm win}-F_{\rm anch}-F_{\rm chunk}:16\mapsto8}_{\text{кадров в кольце}}\end{aligned}$ | Меньше ключей в self-attention. | $\displaystyle \begin{aligned}&\mathrm{FPS}:+12\text{–}15\%,\ \text{чанки 1–3 побитно равны}\\&\underbrace{\lvert\Delta B\rvert>15}_{\text{сдвиг синего, лиса, 12 чанков}}:\ \underbrace{3/12}_{24},\ \underbrace{0/12}_{32}\end{aligned}$<br>Три случая: 608 seed 0, 576 seed 1, 512 seed 0. На $544\times960$ при окне 24 дрейфа нет ($\Delta B$ от $-13$ до $+6$).<br>На смене сцены `cache.rs:135` падает: кольцо равно чанку.<br>[Отчёт, §27–29](docs/report-2026-11-11-loss-704-608-for-fable.md) · [STATE](docs/STATE.md) |
| Не применяем: шов $3.86\ /\ 4.68$<br>$\displaystyle F_{\rm win}:32\mapsto16$ | $\displaystyle 16=\underbrace{8}_{\text{якорь}}+\underbrace{0}_{\text{кольцо}}+\underbrace{8}_{\text{чанк}}$ | Предел уменьшения окна. | $\displaystyle \underbrace{\mathrm{FPS}:+27\%}_{608\times1120}$<br>Чашка пустеет, кадр сбрасывается. |
| Не применяем: $\cos V=0.43$<br>$\displaystyle \mathrm{KVnew}_{n+1}:t=0\mapsto t_2$<br>`cache_write: last_step` | $\displaystyle \begin{aligned}&\text{Было: отдельный проход записи при }t=0.\\&\text{Стало: K/V берём из шага }t_2.\\&T_{\rm chunk}:3\cdot30\,T_{\rm block}\mapsto2\cdot30\,T_{\rm block}\end{aligned}$ | Убрать треть чанка. | $\displaystyle \begin{aligned}&\underbrace{\cos V=0.43,\ \cos K=0.76}_{t_2=833\ \text{к}\ t=0}\\&\underbrace{\mathrm{FPS}:8.60\to11.25}_{608\times1120}\end{aligned}$<br>Вода в чашке не набирается, у лисы уплывает камера.<br>[Отчёт, ч. 3](docs/report-2026-11-11-loss-704-608-for-fable.md) |
| Не применяем: $\cos V\le0.70$<br>$\displaystyle t_2:833\mapsto t$ | $\displaystyle \begin{aligned}&\text{Зашумляем чистый латент до }t\text{, сравниваем K/V.}\\&t\in\{600,400,250,100\}\end{aligned}$ | Есть ли $t$, при котором K/V близки к чистым. | $\displaystyle \cos V:0.48,\ 0.58,\ 0.66,\ 0.70$<br>K/V зависят от модуляции по $t$. |
| Не применяем: $\mathrm{FPS}-1.4\%$<br>$\displaystyle \mathrm{KVvid}_n\mapsto\mathrm{KVanch}$ в записи<br>`clean_pass_sink` | $\displaystyle \begin{aligned}&\text{В проходе записи внимание только на якорь и чанк.}\\&\underbrace{T_{\rm Sage}:12.0\to4.1\ \mathrm{ms}}_{\text{оценка, чанк 1}}\end{aligned}$ | Удешевить третий проход. | Маска даёт свои накладные расходы. |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \Phi_{\rm continue}^{(2)}=\Phi_{\rm continue}^{(1)}$ | $\displaystyle (F_{\rm chunk},F_{\rm anch},F_{\rm win})=(8,8,32),\ \text{запись при }t=0$ | Память видео MVP1 без изменений. | $\displaystyle \underbrace{0/12}_{\text{сильный дрейф цвета}},\quad\underbrace{1.71\ /\ 1.13}_{\text{шов: лиса / чашка, }512}$<br>[Сетки окон](D:/Experiments/DIT/outputs/kv-window-2026-11-11/) |

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Компиляция и размещение · MVP2

Что ограничивает скорость и что ускоряется без изменения результата? Рассмотрим состав одного блока.

$$
\underbrace{T_{\rm block}=24.3\ \mathrm{ms}}_{512\times928,\ F_{\rm win}=32}:\quad
\underbrace{38.6\%}_{\text{линейные W4A4}},\
\underbrace{37.9\%}_{\text{Sage self и cross}},\
\underbrace{7.5\%}_{\text{подготовка Sage}},\
\underbrace{6.8\%}_{\text{нормы, модуляция}},\
\underbrace{3.6\%}_{\text{квантование активаций}},\ 
\underbrace{4.8\%}_{\text{сумма lora, RoPE, запись KV}}
$$

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{Attn}_{\rm cross}:\mathrm{SDPA}\mapsto\mathrm{Sage\,2}$ | $\displaystyle \begin{aligned}&\text{Было в MVP1, в point-b2 убрано, возвращаем с замером.}\\&\underbrace{\mathrm{KVtext}}_{\text{один раз на промпт}}\end{aligned}$ | То же ядро, что в self-attention. | $\displaystyle \begin{aligned}&\underbrace{\mathrm{FPS}:14.18\to14.71,\ \Delta\overline{\mathcal L}_1=+0.001}_{512\times928}\\&\Delta\overline{\mathcal L}_1:\ \underbrace{0.000}_{544\times960},\ \underbrace{+0.022}_{544\times992}\end{aligned}$<br>[Отчёт, §28, §30](docs/report-2026-11-11-loss-704-608-for-fable.md) |
| <span class="wan-kept">Применяем</span> только в Live<br>$\displaystyle \Sigma_{\rm ordered}\mapsto\Sigma_{\rm atomic}$, $B_{\rm VRAM}^{\rm margin}:0.0625\mapsto0$ | $\displaystyle \text{В Live побитный повтор не нужен.}$ | Быстрый порядок сложения и весь запас VRAM. | $\displaystyle \underbrace{14.71\to15.05}_{512\times928},\quad\underbrace{13.08\to13.44}_{544\times960}$<br>$13.08$ в отчёте нет, только $13.44$; конфиг замера [sagecross-fps-544x960-win32.json](configs/experiments/speed-2026-11-12/day2/sagecross-fps-544x960-win32.json). |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{policy}:\mathrm{greedy\_ram}\mapsto\mathrm{greedy\_vram}$ | $\displaystyle \underbrace{7.1\ \mathrm{ms}}_{\text{копия строки RAM}\to\text{VRAM}}<T_{\rm block}$ | Проверяем, ограничивает ли доставка весов. | $\displaystyle \underbrace{1\ \mathrm{ms}}_{\text{ожидание копий за прогон}},\ \underbrace{0}_{\text{опозданий}}$<br>Ограничивает счёт блока.<br>[Отчёт, §12](docs/report-2026-11-11-loss-704-608-for-fable.md) |
| Не применяем: $+2.8\%$ при $\Delta\overline{\mathcal L}=+0.023$<br>$\displaystyle \mathrm{Attn}_{\rm self}\mapsto\mathrm{Attn}_{\rm sparse}$<br>`kernels.sparse` | $\displaystyle \begin{aligned}&\text{Отбрасываем ключи с малым вкладом.}\\&\underbrace{\varepsilon=0.02,\ \text{якорь}+8}_{\text{вариант 1}},\quad\underbrace{\text{якорь}+\text{последние 4}}_{\text{вариант 2, }+1.0\%}\end{aligned}$ | Меньше ключей в Sage. | При окне 24 ключей всего три чанка, отбрасывать мало. |
| Не применяем: $\mathrm{FPS}:13.44\to13.37$<br>$\displaystyle T_{\rm dec}\parallel T_{\rm block}$ | $\displaystyle \begin{aligned}&\text{Декодер на своём потоке поверх следующего чанка.}\\&T_{\rm dec}=8\cdot12.7\ \mathrm{ms}\end{aligned}$ | Перекрыть декодер с блоками. | Латент побитно равен, $M_{\rm VRAM}=8.07$ ГБ. GPU уже занят счётом.<br>[Отчёт, §30](docs/report-2026-11-11-loss-704-608-for-fable.md) |
| Не применяем: $7.3+2.2>8$ ГБ<br>$\displaystyle \mathrm{prep}(K_{\rm win}):3\mapsto1$ на чанк | $\displaystyle \begin{aligned}&\text{Окно прошлых чанков между проходами не меняется,}\\&\text{но }\overline K,\ \operatorname{scale}(V)\text{ считаются по всем ключам с новым чанком.}\end{aligned}$ | Не пересчитывать подготовку окна. Побитно невозможно. | $\displaystyle \underbrace{2.2\ \text{ГБ}}_{\text{кэш окна в VRAM}},\quad\underbrace{\le3.6\%}_{\text{оценка выигрыша}}$ |
| Отложено: ждём итога ядер<br>$\displaystyle \mathrm{Norm}\to\mathrm{Quant}_4\mapsto\mathrm{fused}$ | $\displaystyle \begin{aligned}&\text{Норма, модуляция и квантование одним ядром.}\\&0.86+0.79\ \mathrm{ms}\ \text{на блок}\end{aligned}$ | Убрать запись и чтение BF16. | Оценка $\approx2\%$ (§29), нужно новое ядро в Nunchaku. Времён $0.86+0.79$ мс в отчёте нет. |

$$
\underbrace{T_{\rm chunk}=2.39\ \mathrm{s}}_{544\times960,\ F_{\rm win}=32}=\underbrace{90\cdot25.2\ \mathrm{ms}}_{\text{блоки}}+\underbrace{0.10\ \mathrm{s}}_{\text{декодер}},\qquad\mathrm{FPS}=\frac{32}{2.39}=13.4.
$$

Преобразования с побитно равным латентом исчерпаны. Дальше скорость дают только ядра линейных W4A4 и Sage.

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Неизменные компоненты · MVP2

Что в MVP2 осталось от MVP1? Рассмотрим компоненты, которые этот этап не трогал.

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{DiT}^{(2)}=\mathrm{DiT}^{(1)},\ \mathrm{dec}^{(2)}=\mathrm{dec}^{(1)}$ | $\displaystyle \mathrm{LongLive\,2.0\mbox{-}5B\ S2},\ N_{\rm steps}=2,\ \mathrm{TAEHV}$ | Модель и декодер MVP1. | $\displaystyle \underbrace{T_{\rm dec}=0.10\ \mathrm{s}\ \text{из}\ T_{\rm chunk}=2.39\ \mathrm{s}}_{544\times960}$<br>Не измерено: $\mathcal L$ TAEHV к Wan VAE. |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \Phi_{\rm scene}^{(2)}=\Phi_{\rm scene}^{(1)}$ | $\displaystyle F_{\rm scene\,offset}=8,\ \text{multi-shot sink}$ | Смену сцены этот этап не трогал. | Режим «продолжение или новая сцена» по-прежнему не задан. |
| Отложено: выбор пользователя<br>$\displaystyle W_{\rm T5}^{(2)}=\mathrm{BF16}$ | $\displaystyle \text{MVP1: INT8}\mapsto\text{MVP2: BF16}$ | INT8 даёт быструю смену промпта, в просмотре хуже соответствует. | $T_{\rm prompt}$: $2.86$ с BF16 против $1.63$ с INT8 ([E30](notes/research/cards/E30.%20Стартовая%20точка%20W4A4%20704×1280%20и%20отложенные%20шаги.md)). |

---

**Выбор рабочей точки 544×960, окно 32 · следующий этап**

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Итог · MVP2

Какую конфигурацию закрепляем? Рассмотрим кандидатов рядом с ограничением $\mathrm{FPS}\ge15$.

| Кандидат | FPS, Live | $M_{\rm VRAM}$ | $\overline{\mathcal L}_1$ | $\overline{\mathcal L}_1/\mathcal L_{\rm ceil}$ | Решение |
| :--- | :--- | :--- | :--- | :--- | :--- |
| <span class="wan-kept">Применяем</span><br>$544\times960,\ F_{\rm win}=32$ | $13.44$ | $7.3$ ГБ (режим эксперимента, в Live не записан) | $0.426$ | $0.58$ | Выбор пользователя по сеткам: качество важнее $1.5$ FPS |
| $544\times960,\ F_{\rm win}=24$ | $15.03$ | $6.8$ ГБ | $0.426$ | $0.58$ | Не применяем: падает на смене сцены. На этой точке дрейфа нет, 3 из 12 было на 608, 576, 512 |
| $512\times928,\ F_{\rm win}=32$ | $15.05$ | $7.1$ ГБ | $0.484$ | $0.66$ | Отложено: запасная точка на 15 FPS |

$$
\begin{aligned}
\mathrm{cfg}_{\rm MVP2}&:\quad 544\times960,\quad h\times w=17\times30,\quad\underbrace{M_H=\tfrac{22}{17},\ M_W=\tfrac{40}{30}}_{\text{PI по отношению сеток}},\quad\underbrace{(8,8,32)}_{(F_{\rm chunk},F_{\rm anch},F_{\rm win})},\\
&\underbrace{\mathrm{W4A4},\ \text{T5 BF16},\ \text{KV FP8}}_{\text{точность}},\quad\underbrace{\mathrm{Sage\,2}\ \text{self и cross},\ \mathrm{RoPE}_{\rm LongLive}\ \text{отдельно}}_{\text{ядра}},\quad\underbrace{\Sigma_{\rm atomic},\ \mathrm{greedy\_vram},\ 0\ \text{ГБ}}_{\text{Live}}.
\end{aligned}
$$

| Ограничение | Статус | Число |
| :--- | :--- | :--- |
| $\mathrm{FPS}\ge15$ | Не выполнено | $13.44$, не хватает $11.6\%$ |
| $M_{\rm VRAM}\le8$ ГБ | Выполнено | $7.3$ ГБ в режиме эксперимента ([§28](docs/report-2026-11-11-loss-704-608-for-fable.md)) |
| $M_{\rm RAM}\le16$ ГБ | Не измерено | Пик RAM в отчётах числом не записан. |
| $\overline{\mathcal L}_1$ | $0.426$ | доля зазора $0.58$ |

Ближайшая альтернатива: $512\times928$, $F_{\rm win}=32$, $15.05$ FPS ценой $+0.058$ потери.

Открыто: недостающие $11.6\%$ ищем только в ядрах линейных W4A4 и Sage, разрешение и окно не меняем. T5 BF16 или INT8 решает пользователь.

[Конфиг Live](configs/candidates/544x960-win32-live.json) · [Конфиг эксперимента](configs/candidates/544x960-win32-experiment.json) · [512×928](configs/candidates/512x928-win32-live.json) · [Итоговые сетки](D:/Experiments/DIT/outputs/speed-2026-11-12/grids/) · [Отчёт 11–12.11](docs/report-2026-11-11-loss-704-608-for-fable.md)

Расхождения с конфигами: в матрице версий строка MVP2 ссылается на [configs/mvp2.json](configs/mvp2.json), а там $512\times928$ и `rope.stretch` $=0.299$, то есть не рабочая точка. [configs/mvp1.json](configs/mvp1.json) собран под $544\times960$, но offloading в нём старый (`vram_margin_gib` $=1.5$, без `greedy_vram`). Совпадает с итогом только [544x960-win32-live.json](configs/candidates/544x960-win32-live.json).
