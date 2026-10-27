<span class="wan-scene-style wan-scene-math wan-scene-grid wan-offloading"></span>

# Offloading

Рассмотрим генерацию следующего чанка. В прямоугольниках — необходимые данные и веса.

$$
\underbrace{\boxed{p}}_{\text{промпт}}
\xrightarrow[\boxed{W^{\rm text}}]{\text{T5}}
\underbrace{\boxed{c}}_{\text{текстовое условие}}
$$
$$
\left(\underbrace{\boxed{\eta_{n+1}}}_{\text{шум}},\boxed{c},
\underbrace{\boxed{\mathcal S_n}}_{\text{KV видео}}\right)
\xrightarrow[\boxed{W^{\rm DiT}}]{\text{DiT}}
\left(\underbrace{\boxed{\mathcal S_{n+1}}}_{\text{обновлённый KV self-attention}},
\underbrace{\boxed{Z_{n+1}}}_{\text{латентный чанк}}\right)
$$
$$
\boxed{Z_{n+1}}
\xrightarrow[\boxed{W^{\rm dec}}]{\text{VAE-декодер}}
\underbrace{\boxed{Y_{n+1}}}_{\text{кадры}}
\longrightarrow\mathrm{MP4}
$$
$$
\underbrace{\boxed{c},\ \boxed{\mathcal S_{n+1}}}_{\text{сохраняем между чанками}},\qquad
\underbrace{\boxed{\mathcal T(c)},\ \boxed{Z_{\rm window}},\ \boxed{h^{\rm dec}}}_{\substack{\text{по режиму: KV текста, латенты для recache,}\\\text{состояние потокового декодера}}}.
$$

Объединим это с рабочими буферами и кодом вычислений в ресурсы.

$$
\begin{aligned}
\underbrace{\mathcal R}_{\text{ресурсы}}
={}&\underbrace{\mathcal R_W}_{\text{веса}}
\cup\underbrace{\mathcal R_{\rm state}}_{\text{KV, состояние декодера}}
\cup\underbrace{\mathcal R_x}_{\text{активации, рабочие буферы}}\\[8pt]
&\cup\underbrace{\mathcal R_{\rm data}}_{\text{входы, условия, латенты, кадры}}
\cup\underbrace{\mathcal R_{\rm code}}_{\text{пакеты вычислений, библиотеки}}.
\end{aligned}
$$

Размещаем сохраняемые тензоры в общем кэше:

$$
\mathrm{VRAM}\rightleftarrows
\underbrace{\mathrm{RAM}_{\rm cache}}_{\text{ограниченный бюджет}}
\rightleftarrows\mathrm{SSD}.
$$

Рассмотрим порядок вычислений и ресурсы каждого узла:

$$
\underbrace{G=(V,E)}_{\text{вычисления и зависимости}},\qquad
\underbrace{v_1,\ldots,v_N}_{\text{порядок исполнения}},\qquad
\underbrace{\mathcal U_i\subseteq\mathcal R}_{\text{ресурсы узла }v_i}.
$$

| <span class="wan-offload-methods">Метод / источник</span> | Контекст → вычисление | Идея |
| :--- | :--- | :--- |
| **Граф времени жизни**<br>[ResourcePlan](../../src/wan_dit_lab/offloading/resource_schedule.py)<br>Родственные подходы: [SwapAdvisor](https://par.nsf.gov/servlets/purl/10191573), [TensorShunt](https://github.com/eladwf/TensorShunt/blob/main/python/tensorshunt/schedule.py) | $\displaystyle\begin{aligned}&\underbrace{\mathrm{first}(r)=\min\{i:r\in\mathcal U_i\}}_{\text{первое использование}}\\[6pt]&\underbrace{\mathrm{last}(r)=\max\{i:r\in\mathcal U_i\}}_{\text{последнее использование}}\\[6pt]&\mathrm{acquire}(r)\to v_{\mathrm{first}(r)}\to\cdots\to v_{\mathrm{last}(r)}\to\mathrm{release}(r)\end{aligned}$ | Когда ресурс нужен и когда его можно освободить. Перед использованием ждём загрузку, перед освобождением ждём завершение CUDA. |
| **Окна весов 2×2**<br>[BlockOffloader](../../src/wan_dit_lab/offloading/offload.py)<br>[Diffusers: group offloading](https://huggingface.co/docs/diffusers/optimization/memory#group-offloading) | $\displaystyle\underbrace{2\text{ блока в pinned RAM}}_{\text{подготовка}}\longrightarrow\underbrace{2\text{ блока в VRAM}}_{\text{вычисление}}$ | Исходная реализация для весов. Фиксированная глубина предзагрузки. |
| **Общий кэш ресурсов**<br>[ResourceCache](../../src/wan_dit_lab/offloading/resource_cache.py)<br>**Применяем** | $\displaystyle\begin{aligned}&\mathrm{VRAM}\rightleftarrows\mathrm{RAM}_{\rm cache}\rightleftarrows\mathrm{SSD}\\[6pt]&\mathrm{active}(r)\Rightarrow\text{не вытесняем}\\[6pt]&\mathrm{dirty}(r)\land\mathrm{evict}(r)\Rightarrow\mathrm{write}(r)\end{aligned}$ | Единые бюджеты. Сначала вытесняем ресурс с наиболее далёким следующим использованием; иначе по давности обращения. Чистый файл переиспользуем. |
| **Модель стоимости**<br>[FlexGen §4.2–4.3](https://arxiv.org/html/2303.06865#S4)<br>**Подключена к общему кэшу** | Сначала измеряем:<br>$\displaystyle\underbrace{\Delta t_{\rm prepare}(r),\ \Delta t_{\rm transfer}(r)}_{\text{подготовка и перенос ресурса}},\quad\underbrace{\Delta t_{\rm compute}(v_i)}_{\text{время выполнения узла}}$<br>Затем выбираем:<br>$\displaystyle\underbrace{\pi}_{\text{что, куда и когда переносить}},\qquad\min_{\pi}\widehat t_{\rm end}(\pi)$<br>при бюджетах RAM / VRAM и зависимостях $G$. | Расписание по замерам. Наши профили: медианы; запас = максимум загрузки и минимум вычисления. Ранее подбирали только окна весов. |

$$
\Delta t_{\rm disk}(r)=
\frac{\underbrace{B_{\rm read}(r)}_{\text{прочитанные байты}}}{\underbrace{\beta_{\rm read}}_{\text{байт/с}}}
+
\frac{\underbrace{B_{\rm write}(r)}_{\text{записанные байты}}}{\underbrace{\beta_{\rm write}}_{\text{байт/с}}}.
$$

**Применяем:** общий кэш. В профилях стоимости размещение выбирается целочисленной оптимизацией по замерам Read / Write / восстановления; предзагрузка учитывает время до использования. Это адаптация модели стоимости FlexGen к целым ресурсам, не исходный оптимизатор статьи. Веса пока используют отдельные окна; Предзагрузка до VRAM учитывает очередь переносов; commit выполняется внутри аренды attention. Изменённый ресурс записывается целиком.

[Медианы](../../configs/experiments/offloading_managed_cost_median.json) · [Запас](../../configs/experiments/offloading_managed_cost_guarded.json) · [Реализация](../../src/wan_dit_lab/offloading/resource_cache.py) · [Ограничения](../../docs/managed-resource-cache.md)

<!-- offloading-comparison:start -->

480×832, compile=default. Веса: пять промптов по 30 с; ресурсы: один промпт, 5 с. Прогрев различается, прямое сравнение FPS и пиков ограничено. · = нет замера.

| <span class="wan-offload-results wan-offload-comparison">Метрика</span> | Веса: 2×2 | Ресурсы: текущий |
| :--- | ---: | ---: |
| Роликов | 5 | 1 |
| Длительность видео, с | 30 | 5 |
| torch.compile | default | default |
| Прогрев, чанков | 7 | 3 |
| FPS после прогрева | 1.292 | 1.468 |
| Генерация, с | 403.669 | 71.639 |
| Первый кадр, с | 46.656 | 27.021 |
| Чанк p50, с | 9.305 | 8.215 |
| Чанк p90, с | 9.366 | 9.357 |
| Чанк p99, с | 9.418 | 9.785 |
| VRAM пик тензоров, GiB | 7.005 | 7.003 |
| VRAM пик резерва, GiB | 7.232 | 7.230 |
| VRAM устройства: пик занятой, GiB | 7.595 | 7.593 |
| VRAM устройства: минимум свободной, GiB | 0.401 | 0.403 |
| RAM RSS пик, GiB | 9.916 | 5.625 |
| Private commit пик, GiB | 15.355 | 11.770 |
| Pinned RAM пик, GiB | 1.250 | 0.656 |
| RAM-кэш пик, GiB | · | 0.129 |
| Диск Read, GiB | · | 0.000 |
| Диск Write, GiB | · | 0.008 |
| Диск Read, с | · | 0 |
| Диск Write, с | · | 0.072 |
| Опоздания / обращения к ресурсам | · | 0 / 2366 |

[Полные замеры и ролики](../../outputs/experiments/2026-10-19/22_Calibrated_prefetch_5s/Comparison_current.md) · [Старые исходные данные](../../outputs/experiments/2026-10-18/12_Offloading_full_pipeline_30s/Сравнение.md)

<!-- offloading-comparison:end -->





Кэш ресурсов: VRAM 6.19 GiB, RAM ≤ 832 MiB. Кадры совпадают с исходным прогоном 18.10. На том же коротком префиксе веса 2×2: 1.596 FPS.

[Калибровка и сопоставимый префикс](../../outputs/experiments/2026-10-19/22_Calibrated_prefetch_5s/Comparison.md) · [Текущий профиль](../../configs/experiments/offloading_managed_cost_guarded.json)









**Источники:** [FlexGen §4.2–4.3](https://arxiv.org/html/2303.06865#S4) · [Diffusers: group offloading](https://huggingface.co/docs/diffusers/optimization/memory#group-offloading) · [PyTorch: mmap](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html#using-torch-load-mmap-true).
