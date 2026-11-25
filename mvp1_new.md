<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

# Исследование · MVP1

Переписано 12.11.2026 по стилю research-blackboard-style из [Исследование.canvas](Исследование.canvas) и шести заметок `notes/research/`. Оригиналы не менялись. Числа взяты из оригиналов без пересчёта, у каждого ссылка на карточку E или отчёт. Сверено 16.11.2026 с карточками E01, E02, E04, E05, E10–E15, E22, E24, E26, E29, E30 и заметками.

## Контекст и трудность

$$
\begin{aligned}
\mathrm{prompt}&\xrightarrow{\mathrm{cfg}}\mathrm{video},&&\text{Рассмотрим видеогенерацию по промпту в режиме live}\\
\mathrm{cfg}&\in\mathcal C,&&\text{с конфигурациями из множества C.}\\
\mathrm{FPS}(\mathrm{cfg})&\ge15,&&\text{Режим live требует не меньше 15 кадров в секунду.}\\
M_{\rm VRAM}&\le8\ \text{ГБ},\quad M_{\rm RAM}\le16\ \text{ГБ},&&\text{Трудность: на RTX 4060 Laptop память мала,}\\
T_{\rm block}^{\rm BF16}&\gg\tfrac{1}{15}\ \mathrm{s}\ /\ (\text{кадр}\cdot\text{блок}),&&\text{а обученная модель в BF16 на порядок медленнее показа.}
\end{aligned}
$$

## Цель MVP1

Собрать работающую live-видеогенерацию на RTX 4060 Laptop: выбрать модель, способы ускорения и экономии памяти, продолжение видео и смену сцены. Качество к эталону на этом этапе не измеряется метрикой, только в просмотре.

## Задача

$$
\begin{aligned}
\text{найти}\quad&\mathrm{cfg}\in\mathcal C\\
\text{при условиях}\quad&\mathrm{FPS}(\mathrm{cfg})\ge15,\quad M_{\rm VRAM}(\mathrm{cfg})\le8\ \text{ГБ},\quad M_{\rm RAM}(\mathrm{cfg})\le16\ \text{ГБ},\\
&\text{видео в просмотре связное на 6 и более чанках}.
\end{aligned}
$$

Задача 1: построить модель системы, где каждый компонент это отображение, чтобы отбрасывать конфигурации до опыта. Задача 2 (опыты на $\mathcal L$) начинается в MVP2.

## Общие обозначения

Рассмотрим генерацию видео по промпту чанками.

$$
\underbrace{\mathrm{prompt}_j\xrightarrow{\mathrm{T5}}\mathrm{ctx}_j}_{\text{промпт в векторы, один раз на промпт}},\qquad
\underbrace{\mathrm{KVtext}_j=\mathrm{KV}^{\rm cross}(\mathrm{ctx}_j)}_{\text{K/V текста для cross-attention}},\qquad
\underbrace{\mathrm{lat}_{n+1}\xrightarrow{\mathrm{dec}}\mathrm{frame}_{n+1}}_{\text{латентный чанк в кадры}}
$$
$$
\underbrace{(\epsilon_{n+1},\mathrm{ctx},\mathrm{KVvid}_n)\xrightarrow{\mathrm{DiT}}(\mathrm{lat}_{n+1},\mathrm{KVvid}_{n+1})}_{\text{один чанк, }n=0,\ldots,N-1},\qquad
\underbrace{\mathrm{KVvid}_n=\mathrm{KVanch}\oplus\mathrm{KVupd}_n}_{\text{якорь и обновляемая часть памяти видео}}
$$
$$
\underbrace{F_{\rm chunk},F_{\rm anch},F_{\rm win}}_{\text{латентных кадров: чанк, якорь, окно}},\qquad
\underbrace{H\times W}_{\text{кадр}},\qquad
\underbrace{h\times w=\tfrac{H}{32}\times\tfrac{W}{32}}_{\text{сетка токенов кадра}},\qquad
\underbrace{T_{\rm block}}_{\text{время одного блока DiT}},\qquad
\underbrace{\mathrm{FPS}=\tfrac{4F_{\rm chunk}}{T_{\rm chunk}}}_{\text{4 кадра на латентный кадр}}
$$

Теперь рассмотрим каждый компонент: какую модель берём, как её ускоряем и как продолжаем видео.

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Модель · MVP1

Какие веса и какое правило генерации взять? Рассмотрим правило генерации $\Phi_m$ модели $m\in\{\mathrm{Wan},\mathrm{SF},\mathrm{LL}\}$ (Wan, Self-Forcing, LongLive).

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| Не применяем: ролик ограничен длиной клипа<br>$\displaystyle \Phi_{\rm Wan}:(\epsilon_{0:N-1},\mathrm{ctx})\mapsto\mathrm{lat}_{0:N-1}$<br>[Wan, §4.1–4.2](https://arxiv.org/html/2503.20314v1) | $\displaystyle \begin{aligned}&\text{Рассмотрим генерацию клипа целиком.}\\&\underbrace{\epsilon_{0:N-1}\xrightarrow{\Phi_{\rm Wan}}\mathrm{lat}_{0:N-1}}_{\text{денойзинг всех чанков совместно}}\end{aligned}$ | Отправная точка. Продолжение за пределы клипа требует другого правила. | Не измерено: FPS live для Wan. |
| Не применяем: заменено LongLive<br>$\displaystyle \Phi_{\rm Wan}\mapsto\Phi_{\rm SF}:(\mathrm{KVvid}_n,\mathrm{ctx},\epsilon_{n+1})\mapsto(\mathrm{lat}_{n+1},\mathrm{KVvid}_{n+1})$<br>[Self Forcing, §3.1–3.4](https://arxiv.org/html/2506.08009v1) | $\displaystyle \begin{aligned}&\text{Было: все чанки сразу. Стало: чанк за чанком.}\\&p(\mathrm{lat}_{0:N-1})=\prod_n p(\mathrm{lat}_n\mid\widehat{\mathrm{lat}}_{<n},\mathrm{ctx})\\&\underbrace{\widehat{\mathrm{lat}}_{<n}}_{\text{при обучении сгенерированы самой моделью}}\end{aligned}$ | Авторы: обучение на своих же выходах снимает разрыв обучения и применения. Наше: готовые веса DMD, кольцо KV, FIFO и PC. | $\displaystyle \begin{aligned}&\underbrace{(F_{\rm anch},F_{\rm win}):(10,21)\to(3,12)}_{\text{E10, 480p, веса те же}}\\&\mathrm{FPS}:1.42\to1.86,\quad M_{\rm VRAM}^{\rm peak}:7.0\to4.6\ \mathrm{GiB}\end{aligned}$<br>[E10](notes/research/cards/E10.%20Окно%20LongLive%2012%20кадров,%20скорость%20480p.md) |
| Не применяем: заменено 2.0-5B<br>$\displaystyle W_{\rm SF}\mapsto W_{\rm LL\,1.3B}$<br>[LongLive, §3](https://arxiv.org/html/2509.22622v1) | $\displaystyle \begin{aligned}&\text{Правило то же, }\Phi_{\rm LL}=\Phi_{\rm SF}\text{, меняются веса.}\\&\underbrace{\text{якорь и короткое окно}}_{\text{входят в обучение длинным продолжениям}}\\&(F_{\rm chunk},F_{\rm anch},F_{\rm win})=(3,3,12)\end{aligned}$ | Авторы обучают с окном и якорем, поэтому короткое окно не ломает сцену. | $\displaystyle \begin{aligned}&\underbrace{\mathrm{FPS}=1.75\text{–}1.93}_{480\times832,\ \text{E11}}\\&\underbrace{60\ \mathrm{s}}_{\text{E12}}:\ \text{в просмотре дрейфа нет, на 3-м промпте второй кот}\end{aligned}$<br>[E11](notes/research/cards/E11.%20LongLive-1.3B%20смена%20сцены,%20480p%2030%20с.md) · [E12](notes/research/cards/E12.%20LongLive-1.3B%20показательные%20промпты,%2060%20и%2030%20с.md) · [H14](notes/research/cards/H14.%20Полный%20LongLive%20лучше%20соответствует%20демонстрации.md) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle W_{\rm LL\,1.3B}\mapsto W_{\rm LL\,2.0\,5B\,S2}$<br>[LongLive 2.0](https://arxiv.org/abs/2605.18739) | $\displaystyle \begin{aligned}&\text{Рассмотрим готовые веса на два шага.}\\&\underbrace{N_{\rm steps}:4\mapsto2}_{\text{дистиллированный вариант S2 авторов}}\\&(F_{\rm chunk},F_{\rm anch},F_{\rm win})=(8,8,32),\quad\mathrm{shift}=5\end{aligned}$ | Авторы: прямое AR-дообучение Wan 2.2. Наше: выбор окна и перенос multi-shot sink (см. «Смена сцены»). | $\displaystyle \begin{aligned}&\underbrace{\mathrm{FPS}:4.6\to16.4}_{\text{E26: 2 шага, W4A4, Sage 2 вместе}}\\&\underbrace{480\times832,\quad M_{\rm GPU}^{\rm peak}=5.83\ \mathrm{GiB}}_{\text{6 чанков, в просмотре связно}}\end{aligned}$<br>E26 это проверка без окна Live и без смены сцены, скриптом `live_video.py`.<br>[E14](notes/research/cards/E14.%20LongLive%202%20против%201.3B%20на%20показательных%20промптах.md) · [E26](notes/research/cards/E26.%20Live%20W4A4%20и%20Sage%202,%20480×832.md) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{T5}=\mathrm{UMT5\mbox{-}XXL}$<br>[Wan, §4.1–4.2](https://arxiv.org/html/2503.20314v1) | $\displaystyle \begin{aligned}&\text{Рассмотрим текстовое условие.}\\&\underbrace{\mathrm{prompt}_j\xrightarrow{\mathrm{T5}}\mathrm{ctx}_j}_{\text{вне цикла чанков}}\end{aligned}$ | Цена T5 это задержка смены промпта $T_{\rm prompt}$, а не FPS. VRAM энкодер получает последним. | Точность T5 в карточке «Квантизация». |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{dec}:\ \mathrm{Wan\ VAE}\mapsto\mathrm{TAEHV}$<br>[TAEHV](https://github.com/madebyollin/taehv) | $\displaystyle \begin{aligned}&\text{Рассмотрим декодирование чанка после DiT.}\\&\underbrace{F_{\rm chunk}=8\mapsto32\ \text{кадра}}_{\text{потоковый декодер со своим состоянием}}\end{aligned}$ | Малый декодер, чтобы цену чанка задавал DiT. Латент и KV от декодера не зависят. | $\displaystyle \begin{aligned}&\text{H08: голубой снег у обоих декодеров.}\\&\text{Не измерено: }\mathcal L\text{ TAEHV к Wan VAE.}\end{aligned}$<br>[H08](notes/research/cards/H08.%20Синий%20снег%20вызван%20только%20TAEHV.md) · [E13](notes/research/cards/E13.%20LongLive-2.0-5B%20на%208%20ГБ,%20проба%20разрешений.md) |

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Квантизация · MVP1

Какую точность дать весам и памяти? Рассмотрим квантизацию как отображение тензоров.

$$
\underbrace{\mathcal Q:(W_{\rm DiT},W_{\rm T5},\mathrm{KV})\mapsto(\widehat W_{\rm DiT},\widehat W_{\rm T5},\widehat{\mathrm{KV}})}_{\text{точность весов DiT, T5 и памяти видео}},\qquad
\underbrace{M_X=\lvert X\rvert\,b/8}_{\text{байты при }b\text{ битах, без масштабов}}
$$

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| Не применяем: заменено W4A4<br>$\displaystyle W_{\rm DiT}^{\rm BF16}\mapsto\widehat W_{\rm DiT}^{\rm FP8}$<br>[FP8 Formats, 2022](https://arxiv.org/abs/2209.05433) | $\displaystyle \begin{aligned}&\text{Рассмотрим веса линейных слоёв.}\\&b:16\mapsto8\end{aligned}$ | Память пополам ценой ошибки представления. | $\displaystyle \begin{aligned}&\underbrace{M^{\rm peak}=2.73\ \text{ГиБ}}_{\text{256p, Self-Forcing, веса и KV в FP8}}\end{aligned}$<br>Сравнение деталей с BF16 в E01 не записано.<br>[E01](notes/research/cards/E01.%20FP8%20три%20расписания,%2060%20секунд.md) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \widehat W_{\rm DiT}^{\rm FP8}\mapsto\widehat W_{\rm DiT}^{\rm W4A4}$<br>[SVDQuant (Nunchaku)](https://arxiv.org/abs/2411.05007) | $\displaystyle \begin{aligned}&\text{Было: 8 бит весов, активации BF16.}\\&\text{Стало: 4 бита весов и активаций и ветка низкого ранга.}\\&\underbrace{\mathrm{LongLive\,2.0\ S2}}_{\text{авторы дистиллировали в NVFP4}}\end{aligned}$ | Взято: ядра Nunchaku. Наше: слияния QKV и FFN (карточка «Компиляция»). Не проверено: совпадает ли наш W4A4 SVDQuant с форматом NVFP4 обучения (E30). | $\displaystyle \underbrace{\mathrm{FPS}:6.28\to11.29}_{512\times928,\ \text{E30}}$<br>[E30](notes/research/cards/E30.%20Стартовая%20точка%20W4A4%20704×1280%20и%20отложенные%20шаги.md) · [E22](notes/research/cards/E22.%20W4A4%20SVDQuant%20на%20ядрах%20Nunchaku,%20704×1280.md) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle W_{\rm T5}^{\rm BF16}\mapsto\widehat W_{\rm T5}^{\rm INT8}$ | $\displaystyle \begin{aligned}&\text{Рассмотрим веса текстового энкодера.}\\&b:16\mapsto8,\quad\text{цикл чанков не меняется}\end{aligned}$ | Быстрее смена промпта. | $\displaystyle \begin{aligned}&\underbrace{T_{\rm prompt}:2.86\to1.63\ \mathrm{s}}_{704\times1280,\ \text{8 чанков}}\\&\text{В просмотре промпт соответствует хуже. Метрикой не измерено.}\end{aligned}$<br>[E30](notes/research/cards/E30.%20Стартовая%20точка%20W4A4%20704×1280%20и%20отложенные%20шаги.md) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{KV}^{\rm BF16}\mapsto\widehat{\mathrm{KV}}^{\rm FP8\ E4M3}$ | $\displaystyle \begin{aligned}&\text{Рассмотрим память видео }\mathrm{KVvid}_n.\\&b:16\mapsto8\end{aligned}$ | Память видео растёт с окном, поэтому её квантуем первой. | $\displaystyle \underbrace{M_{\rm KV}:9.819\to4.985\ \mathrm{GiB}}_{704\times1280}$<br>[bf16-native](docs/bf16-native.md) |

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Компиляция · MVP1

Какие ядра и слияния дают время блока? Рассмотрим вычислительный граф одного блока DiT.

$$
\underbrace{G=(V,E)}_{\text{операции и зависимости}},\qquad
\underbrace{\mathcal O:G\mapsto G'}_{\text{выбор ядер и слияния}},\qquad
\underbrace{G'\xrightarrow{\rm torch.export}G_{\rm FX}\xrightarrow{\rm AOTInductor}\mathrm{package}}_{\text{сборка пакета}}
$$

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{Attn}_{\rm self},\mathrm{Attn}_{\rm cross}:\mathrm{SDPA}\mapsto\mathrm{Sage\,2}$<br>[SageAttention 2](https://arxiv.org/abs/2411.10958) | $\displaystyle \begin{aligned}&\text{Рассмотрим self- и cross-attention блока.}\\&\text{Было: }QK^{\top}\text{ и }PV\text{ в BF16.}\\&\text{Стало: }Q,K\text{ в INT8, }PV\text{ в FP8.}\end{aligned}$ | Взято у авторов: ядро sm89. | $\displaystyle \underbrace{T_{\rm block}:186.8\to77.3\ \mathrm{ms}}_{704\times1280,\ \mathrm{FP8},\ \text{только self, E29}}$<br>Не измерено в MVP1: вклад cross отдельно (замер в MVP2).<br>[E29](notes/research/cards/E29.%20Вершина%20A′%20и%20спуск%20B,%20FP8,%20704×1280.md) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle (W_Q,W_K,W_V)\mapsto W_{\rm QKV}$ | $\displaystyle \begin{aligned}&\text{Рассмотрим три проекции входа }X.\\&W_{\rm QKV}^{\top}=[W_Q^{\top}\ W_K^{\top}\ W_V^{\top}]\\&(Q,K,V)=\mathrm{split}(XW_{\rm QKV}^{\top}+b_{\rm QKV})\end{aligned}$ | Наше: один GEMM вместо трёх, вход квантуется один раз. | $\displaystyle \underbrace{T_{\rm QKV}:4.32\to3.77\ \mathrm{ms}}_{\text{блок 0, 7040 токенов, E24}}$<br>[E24](notes/research/cards/E24.%20Слитые%20QKV%20и%20FFN%20в%20W4A4,%20RoPE%20во%20float32,%20704×1280.md) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{FFN}\mapsto\mathrm{FFN}_{\rm fused}$ | $\displaystyle \begin{aligned}&\text{Было: }\mathrm{GEMM}_1\to\mathrm{GELU}\to\mathrm{Quant}_4\text{ тремя ядрами.}\\&\text{Стало: одним эпилогом }\mathrm{GEMM}_1\text{, затем }\mathrm{GEMM}_2.\end{aligned}$ | Наше: промежуточный hidden не пишется в 16 битах. | $\displaystyle \underbrace{T_{\rm FFN}:12.70\to10.67\ \mathrm{ms}}_{\text{блок 0, 7040 токенов, E24}}$<br>[ops/w4a4_ops.cpp](ops/w4a4_ops.cpp) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{RoPE}\overset{\rm fuse}{\hookrightarrow}\mathrm{Attn}_{\rm Sage}$ | $\displaystyle \begin{aligned}&\text{Было: }\mathrm{RMSNorm}\to\mathrm{RoPE}\to\mathrm{Quant}^{Q,K}_{\rm INT8}\text{ отдельно.}\\&\text{Стало: поворот при подготовке INT8 для Sage.}\end{aligned}$ | Наше: убрать отдельный проход RoPE. | Не измерено в MVP1: выигрыш FPS. Статистики RMS считаются отдельным проходом.<br>[sage_rope.py](src/wan_dit_lab/kernels/sage_rope.py) |

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Смена разрешения и RoPE · MVP1

Как сжать кадр с обученного $704\times1280$ без перекоса позиций? Рассмотрим частоты RoPE одной пространственной оси $a\in\{H,W\}$.

$$
\underbrace{M_a=\frac{h}{h'}}_{\text{обученная длина оси к новой}},\qquad
\underbrace{M_H=\tfrac{22}{16}=1.375}_{512\times928},\qquad
\underbrace{\theta_j=b^{-2j/\lvert D\rvert}}_{\text{частоты RoPE оси}},\qquad
\underbrace{\lambda_j=2\pi/\theta_j}_{\text{длина волны}}
$$

Все статьи ниже решают расширение, $M<1$. Перенос на сжатие, $M>1$, это наша гипотеза.

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| Не применяем: в MVP1 $\mathrm{shift}=5$<br>$\displaystyle \mathrm{shift}\mapsto\sqrt{\tfrac{H'W'}{HW}}\,\mathrm{shift}$<br>[Scaling Rectified Flow, (23)](https://arxiv.org/html/2403.03206#S5.SS3.SSS2) | $\displaystyle \begin{aligned}&\text{Рассмотрим расписание шума }\sigma(u)=\tfrac{\mathrm{shift}\,u}{1+(\mathrm{shift}-1)u}.\\&(480,832,5)\mapsto(256,448,2.68)\end{aligned}$ | Меньше токенов, меньше shift. Это выбор расписания, не дообучение. | $\displaystyle \underbrace{\rho:\ 1.824\ \text{с на 1 с видео}}_{\text{Self-Forcing, BF16, }256\times448}$<br>[E02](notes/research/cards/E02.%20BF16%20три%20расписания,%2060%20секунд.md) · [H06](notes/research/cards/H06.%20BF16%20с%20shift%202.68%20пригоден%20как%20текущий%20кандидат.md) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{pos}\mapsto M_a\,\mathrm{pos}$<br>PI, `rope.stretch` $=1$<br>[Position Interpolation, §2.3](https://arxiv.org/html/2306.15595) | $\displaystyle \begin{aligned}&\text{Было: позиции }0,\ldots,h'-1.\\&\text{Стало: те же позиции, растянутые до }0,\ldots,h-1.\\&\underbrace{\mathrm{pos}\mapsto M^{t}\mathrm{pos}}_{\texttt{rope.stretch}=t}\end{aligned}$ | Позиции остаются в обученном диапазоне. Один множитель на все частоты. | $\displaystyle \begin{aligned}&\underbrace{M\in\{0.73,\ldots,1.49\}}_{\text{космонавт, W4A4, }512\times928}\\&\text{В просмотре 04.11: лучше }M=1.05\text{–}1.1,\ \text{хуже }1.375\end{aligned}$<br>[[Позиции RoPE]] |
| Не применяем: частный случай by-parts<br>$\displaystyle b\mapsto b\,M^{-\lvert D\rvert/(\lvert D\rvert-2)}$<br>[YaRN, App. A.2](https://arxiv.org/html/2309.00071) | $\displaystyle \begin{aligned}&\text{Было: общий множитель позиций.}\\&\text{Стало: высокие частоты как есть, низкие как PI.}\\&b'_H=7158,\quad b'_W=7134\end{aligned}$ | NTK-aware: гладкий переход по частотам. | Отдельно не проверяли. Закрыто в [MVP2](mvp2_new.md). |
| Не применяем: закрыто в MVP2<br>$\displaystyle \theta_j\mapsto\bigl(1-\gamma_j\bigr)M\theta_j+\gamma_j\theta_j$<br>[YaRN, §3.2](https://arxiv.org/html/2309.00071) | $\displaystyle \begin{aligned}&\text{Рассмотрим число волн на оси }\mathrm{waves}_j=h/\lambda_j.\\&\gamma_j=\mathrm{clip}\bigl(\tfrac{\mathrm{waves}_j-\alpha}{\beta-\alpha},0,1\bigr)\\&\mathrm{waves}_j=3.5,\,2.3,\,1.5,\,0.9,\ldots\end{aligned}$ | NTK-by-parts: частоты с многими волнами не трогаем, с меньше чем одной растягиваем как PI. | $\displaystyle \underbrace{10.9\ /\ 10.8\ \mathrm{FPS}}_{(\alpha,\beta)=(1,32),\ (0.25,2)}$<br>Оценка 04.11 не дана. В MVP2 лучшее $M$ не растёт со сжатием. |
| Не применяем: подобрано при $M<1$<br>$\displaystyle \tfrac{q^{\top}k}{\sqrt{\lvert D\rvert}}\mapsto\tfrac{q^{\top}k}{t\sqrt{\lvert D\rvert}}$<br>[YaRN, §3.3](https://arxiv.org/html/2309.00071) | $\displaystyle \begin{aligned}&\sqrt{1/t}=1-0.1\ln M\\&\text{у нас }1/t=0.937\end{aligned}$ | Поправка температуры attention. | Не проверяли: вне области подбора, требует правки ядра Sage. |
| Не применяем: дефекта нет<br>$\displaystyle \theta_k\mapsto\tfrac{2\pi M}{h}$<br>[RIFLEx, §3.2–3.3](https://arxiv.org/html/2502.15894) | $\displaystyle \begin{aligned}&\text{Рассмотрим частоту }k\text{ с периодом у места повтора.}\\&\text{Меняем только её, остальные }\theta_j\text{ как есть.}\end{aligned}$ | Лечит повтор сюжета при удлинении. | У нас повтора нет: периоды $6.3,\,9.7,\,15,\,23,\ldots$ |

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Offloading · MVP1

Как уместить веса и память видео в 8 ГБ без потери FPS? Рассмотрим порядок вычислений и ресурсы каждого узла.

$$
\underbrace{v_1,\ldots,v_N}_{\text{узлы графа }G\text{ в порядке исполнения}},\qquad
\underbrace{\mathcal R_i\subseteq\mathcal R=\mathcal R_W\cup\mathcal R_{\rm state}\cup\mathcal R_x\cup\mathcal R_{\rm data}\cup\mathcal R_{\rm code}}_{\text{ресурсы узла: веса, KV, активации, данные, код}},\qquad
\underbrace{\mathrm{VRAM}\rightleftarrows\mathrm{RAM}\rightleftarrows\mathrm{SSD}}_{\text{уровни размещения}}
$$

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| Не применяем: заменено общим кэшем<br>Окна весов $2\times2$<br>[Diffusers: group offloading](https://huggingface.co/docs/diffusers/optimization/memory#group-offloading) | $\displaystyle \underbrace{2\text{ блока в pinned RAM}}_{\text{подготовка}}\to\underbrace{2\text{ блока в VRAM}}_{\text{вычисление}}$ | Фиксированная глубина предзагрузки, только веса. | [offload.py](src/wan_dit_lab/offloading/offload.py) |
| <span class="wan-kept">Применяем</span><br>Граф времени жизни<br>Реализация своя; родственный подход: [SwapAdvisor](https://par.nsf.gov/servlets/purl/10191573) | $\displaystyle \begin{aligned}&\mathrm{first}(r)=\min\{i:r\in\mathcal R_i\},\quad\mathrm{last}(r)=\max\{i:r\in\mathcal R_i\}\\&\mathrm{acquire}(r)\to v_{\mathrm{first}(r)}\to\cdots\to v_{\mathrm{last}(r)}\to\mathrm{release}(r)\end{aligned}$ | Когда ресурс нужен и когда его можно отдать. | [resource_schedule.py](src/wan_dit_lab/offloading/resource_schedule.py) |
| <span class="wan-kept">Применяем</span><br>Общий кэш ресурсов | $\displaystyle \begin{aligned}&\text{Было: отдельные окна для весов.}\\&\text{Стало: единые бюджеты VRAM и RAM для всех }r.\\&\mathrm{active}(r)\Rightarrow\text{не вытесняем},\quad\mathrm{dirty}(r)\land\mathrm{evict}(r)\Rightarrow\mathrm{write}(r)\end{aligned}$ | Наше: вытесняем ресурс с самым далёким следующим использованием. | [resource_cache.py](src/wan_dit_lab/offloading/resource_cache.py) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{plan}^{\star}\in\arg\min\widehat t_{\rm end}(\mathrm{plan})$<br>[FlexGen, §4.2–4.3](https://arxiv.org/html/2303.06865#S4) | $\displaystyle \begin{aligned}&\text{Измеряем }\Delta t_{\rm transfer}(r),\ \Delta t_{\rm compute}(v_i).\\&\text{Выбираем, что, куда и когда переносить, при бюджетах.}\end{aligned}$ | Авторы: модель стоимости. Наше: медианы профиля, запас по максимуму загрузки и минимуму счёта. | $\displaystyle \underbrace{\mathrm{policy}_1=\mathrm{greedy\_ram},\ B_{\rm VRAM}^{\rm margin}=1.5\ \mathrm{GiB},\ B_{\rm T5}^{\rm first}=1.0\ \mathrm{GiB}}_{\text{матрица версий}}$<br>[H12](notes/research/cards/H12.%20Подгрузка%20блоков%20даст%20полноразмерный%20BF16%20эталон.md) · [E16](notes/research/cards/E16.%20LongLive%202%20калибровка%20offloading%20и%20Live%20FP8.md) |

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Продолжение видео · MVP1

Как продолжать видео при ограниченной памяти без дрейфа? Рассмотрим обновление памяти видео одного блока.

$$
\underbrace{\Phi_{\rm continue}:(\epsilon_{n+1},\mathrm{ctx},\mathrm{KVvid}_n)\mapsto(\mathrm{lat}_{n+1},\mathrm{KVvid}_{n+1})}_{\text{правило продолжения}},\qquad
\underbrace{\mathrm{KVnew}_{n+1}}_{\text{K/V нового чанка}},\qquad
\underbrace{\mathrm{IDX}_{\rm evict}\subseteq\mathrm{rows}(\mathrm{KVupd}_n)}_{\text{вытесняемые строки}}
$$

Опыты до LongLive шли на Self-Forcing, $(F_{\rm chunk},F_{\rm win})=(3,21)$, 480p.

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \mathrm{KVvid}_n\mapsto\mathrm{KVanch}\oplus\mathrm{KVupd}_n$<br>[Deep Forcing, §4.2](https://arxiv.org/html/2512.05081v1#S4.SS2) | $\displaystyle \begin{aligned}&\text{Было: окно без постоянной части.}\\&\text{Стало: первые }F_{\rm anch}\text{ кадров не вытесняются.}\\&\underbrace{21=10+8+3}_{\text{якорь, кольцо, новый чанк}}\end{aligned}$ | Якорь держит опору на исходную сцену. Авторы: якорь около половины окна. | $\displaystyle \begin{aligned}&\underbrace{F_{\rm anch}\in\{0,3,6,9\}}_{\text{в просмотре: 9 понравился, дрейф остаётся}},\quad\text{затем }\{12,15,18\}\\&\underbrace{10\text{ против }14}_{\text{половина окна и конфиг авторов}}:\ \text{оставили }10\end{aligned}$<br>[E05](notes/research/cards/E05.%20Якорь%20и%20PC.md) · [H05](notes/research/cards/H05.%20Якорь%20и%20PC%20устраняют%20весь%20дрейф.md) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle k_i\mapsto\mathrm{Rot}(\omega\Delta_i)\,k_i$<br>Temporal RoPE<br>[Deep Forcing, §4.2](https://arxiv.org/html/2512.05081v1#S4.SS2) | $\displaystyle \begin{aligned}&\text{Рассмотрим временную пару координат ключа.}\\&\underbrace{\Delta_i=\mathrm{pos}_i^{\rm new}-\mathrm{pos}_i^{\rm write}}_{\text{сдвиг к позиции в окне}}\\&\text{Поворачиваем копию, сохранённый }K\text{ не меняем.}\end{aligned}$ | Временное расстояние до якоря ограничено окном. Наше: поворот копии, чтобы ошибка округления не копилась. | $\displaystyle \underbrace{14/14}_{\text{CPU-тесты KV-кэша}}$<br>Отдельного A/B нет. |
| Применяли на Self-Forcing; в конфигах Live ключа PC нет<br>$\displaystyle \mathrm{IDX}_{\rm evict}:\ \text{старейшие}\mapsto\operatorname{Bottom}_{N_{\rm new}}(\mathrm{score})$<br>PC, [Deep Forcing, §4.3](https://arxiv.org/html/2512.05081v1#S4.SS3) | $\displaystyle \begin{aligned}&\text{Было (FIFO): вытесняем старейшие строки }\mathrm{KVupd}_n.\\&\text{Стало (PC): строки с наименьшей связью с новым чанком,}\\&\mathrm{score}_i=\textstyle\sum_q\langle Q_q,\widetilde k_i\rangle\text{ на первом шаге денойзинга.}\end{aligned}$ | Авторы: сохранять строки, нужные новому чанку. Якорь не трогаем в обоих случаях. | $\displaystyle \begin{aligned}&\underbrace{5:11\ \text{(PC : FIFO)},\quad p=0.210}_{\text{16 пар, MUSIQ, }480\times832,\ \text{BF16}}\\&H_0\text{ не отвергаем; по кадрам результат зависит от сцены}\end{aligned}$<br>В [[Длительная генерация]] строка PC помечена «Применяем».<br>[Подробно](notes/research/Длительная%20генерация.md) · [pc_fifo_480p.json](configs/experiments/pc_fifo_480p.json) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \Phi_{\rm continue}^{(1)}=\Phi_{\rm continue}\big\rvert_{(8,8,32)}$ | $\displaystyle \begin{aligned}&\text{LongLive 2.0, FIFO, кольцо в фиксированном буфере.}\\&(F_{\rm chunk},F_{\rm anch},F_{\rm win})=(8,8,32)\end{aligned}$ | Окно и якорь входят в обучение LongLive. | См. итог модели, E26.<br>[causal.py](src/wan_dit_lab/generation/causal.py) |

---

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Смена сцены · MVP1

Как сменить промпт: продолжить сцену или начать новую? Рассмотрим пару (память видео, K/V текста) при переходе $\mathrm{prompt}_0\mapsto\mathrm{prompt}_1$.

$$
\underbrace{\Phi_{\rm scene}:(\mathrm{KVvid}_0,\mathrm{KVtext}_0,\mathrm{prompt}_1)\mapsto\begin{cases}(\varnothing,\mathrm{KVtext}_1),&\text{новая сцена},\\\Phi(\mathrm{KVvid}_0,\mathrm{KVtext}_0),&\text{продолжение}\end{cases}}_{\text{гибридное управление}}
$$

Опыты swap, recache, decay, blend: один ролик, Self-Forcing. Это наблюдения, не сравнение.

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| Отложено: нужно сравнение на нескольких роликах<br>$\displaystyle \Phi_{\rm swap}:\mathrm{KVtext}_0\mapsto\mathrm{KVtext}_1$<br>[LongLive, §4.4](https://arxiv.org/html/2509.22622v2#S4.SS4) | $\displaystyle \begin{aligned}&\text{Заменяем K/V текста, }\mathrm{KVvid}_0\text{ сохраняем.}\end{aligned}$ | Новый текст, прежняя память видео. | В просмотре от recache не отличается.<br>[E04](notes/research/cards/E04.%20Смена%20промпта%20четыре%20способа.md) |
| Отложено: то же<br>$\displaystyle \Phi_{\rm recache}:\mathrm{KVvid}_0\mapsto\mathrm{KVvid}_1$<br>[LongLive, §3.1](https://arxiv.org/html/2509.22622v2#S3.SS1) | $\displaystyle \begin{aligned}&\text{Было: }\mathrm{KVvid}_0\text{ записана при }\mathrm{ctx}_0.\\&\text{Стало: проход готовых }\mathrm{lat}\text{ при }t=0\text{ с }\mathrm{ctx}_1\\&\text{заново пишет }\mathrm{KVvid}_1.\end{aligned}$ | Те же кадры, память видео под новый текст. | В просмотре от swap не отличается.<br>[H03](notes/research/cards/H03.%20Recache%20без%20дообучения%20достаточно%20для%20перехода.md) |
| Не применяем: изображение плывёт<br>$\displaystyle \Phi_{\rm decay}:\mathrm{KVvid}_0\mapsto D(\mathrm{age})\,\mathrm{KVvid}_0$<br>[Echo-Forcing](https://arxiv.org/abs/2605.16003) | $\displaystyle \begin{aligned}&\text{Рассмотрим вес старой строки }w_i=e^{-\mathrm{age}\cdot\mathrm{decay}_i}.\\&\underbrace{\mathrm{decay}_i}_{\text{растёт с }1-\cos(k_i^{\rm old},k^{\rm ref}_{\operatorname{patch}(i)})}\\&D=\operatorname{diag}w\text{ к копиям }K_0,V_0\text{ перед attention.}\end{aligned}$ | Наше: чем сильнее старый токен отличается от новой сцены, тем быстрее забываем. | Причина «плывёт» не установлена, ошибка реализации остаётся гипотезой.<br>[Подробно](notes/research/Смена%20сцены.md) · [H04](notes/research/cards/H04.%20Текущий%20decay%20даёт%20приемлемую%20смену%20сцены.md) |
| Отложено: один ролик<br>$\displaystyle \Phi_{\rm blend}:\mathrm{KVtext}_0\mapsto(1-\alpha_r)\mathrm{KVtext}_0+\alpha_r\mathrm{KVtext}_1$ | $\displaystyle \begin{aligned}&\alpha_r=\min(r/N_{\rm blend},1),\quad r=1,\ldots,N_{\rm blend}\\&\mathrm{KVvid}_0\text{ не пересчитываем.}\end{aligned}$ | Наше: плавная смена текста при той же памяти видео. | В просмотре переход нравится. |
| <span class="wan-kept">Применяем</span><br>$\displaystyle \Phi_{\rm scene}^{(1)}=\Phi_{\rm scene}\big\rvert_{F_{\rm scene\,offset}=8}$<br>multi-shot sink | $\displaystyle \begin{aligned}&\text{Было: якорь из первых кадров ролика.}\\&\text{Стало: первый чанк новой сцены становится якорем}\\&\text{до следующей склейки, позиции сдвигаются на 8 кадров.}\end{aligned}$ | Взято у LongLive 2.0: shot sink и narrative RoPE. Режим «продолжение или новая сцена» не задан. | $\displaystyle \underbrace{\text{смена }1.7\text{–}2.5\ \mathrm{s},\ 2.9\ \mathrm{FPS}}_{480\times832,\ \mathrm{shift}=3.33,\ \text{лиса, пляж, лиса}}$<br>Это не конфигурация MVP1: другой shift, FPS до ускорений.<br>[E15](notes/research/cards/E15.%20LongLive%202%20склейки%20с%20multi-shot%20sink.md) |

---

**Выбор LongLive 2.0-5B S2 и сборка MVP1 · следующий этап**

<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

## Итог · MVP1

$$
\begin{aligned}
\mathrm{cfg}_{\rm MVP1}&:\quad\mathrm{DiT}=\mathrm{LongLive\,2.0\mbox{-}5B\ S2},\quad N_{\rm steps}=2,\quad\mathrm{shift}=5,\quad\mathrm{dec}=\mathrm{TAEHV},\\
&\underbrace{W_{\rm DiT}=\mathrm{W4A4},\quad W_{\rm T5}=\mathrm{INT8},\quad\mathrm{KV}=\mathrm{FP8}}_{\text{квантизация}},\quad
\underbrace{\mathrm{Sage\,2},\ W_{\rm QKV},\ \mathrm{FFN}_{\rm fused},\ \mathrm{RoPE}\hookrightarrow\mathrm{Sage}}_{\text{компиляция}},\\
&\underbrace{512\times928,\ \mathrm{PI},\ t=1}_{\text{разрешение}},\quad
\underbrace{(F_{\rm chunk},F_{\rm anch},F_{\rm win})=(8,8,32)}_{\text{продолжение}},\quad
\underbrace{F_{\rm scene\,offset}=8}_{\text{смена сцены}},\quad
\underbrace{\mathrm{greedy\_ram},\ 1.5\ \mathrm{GiB}}_{\text{offloading}}.
\end{aligned}
$$

| Ограничение | Статус | Число |
| :--- | :--- | :--- |
| $\mathrm{FPS}\ge15$ | На $480\times832$ выполнено только в проверке без окна Live; на $512\times928$ не выполнено | $16.4$ ([E26](notes/research/cards/E26.%20Live%20W4A4%20и%20Sage%202,%20480×832.md)), около $14.5$ в самой конфигурации MVP1 (только чанки 1–2, на 3-м OOM) и $11.29$ при T5 BF16 и раздельном RoPE ([E30](notes/research/cards/E30.%20Стартовая%20точка%20W4A4%20704×1280%20и%20отложенные%20шаги.md)) |
| $M_{\rm VRAM}\le8$ ГБ | Выполнено на $480\times832$, не выполнено на $512\times928$ | $M_{\rm GPU}^{\rm peak}=5.83$ GiB на $480\times832$, OOM на 3-м чанке на $512\times928$ |
| $M_{\rm RAM}\le16$ ГБ | Не измерено отдельно | |
| $\mathcal L$ | Не измерено | Метрика появилась в MVP2: $512\times928$ даёт $\overline{\mathcal L}_1=0.483$ при T5 BF16 и раздельном RoPE |

Ближайшая альтернатива: LongLive-1.3B, $1.75$–$1.93$ FPS на $480\times832$ без W4A4 и Sage 2 ([E11](notes/research/cards/E11.%20LongLive-1.3B%20смена%20сцены,%20480p%2030%20с.md)), до ограничения далеко.

Открыто: режим смены сцены не выбран, точность T5 спорна (в просмотре INT8 хуже), $\mathcal L$ не измерена, PI с $t=1$ в просмотре хуже $M=1.05$–$1.1$.

Расхождение с конфигом: [configs/mvp1.json](configs/mvp1.json) с 12.11 собран под $544\times960$ (рабочая точка MVP2) и в git не закоммичен, поэтому конфига MVP1 в чистом виде на диске нет. Последняя закоммиченная версия: `git show 992bf0c:configs/mvp1.json`.
