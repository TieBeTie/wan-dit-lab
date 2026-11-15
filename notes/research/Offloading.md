<span class="wan-scene-style wan-scene-math wan-scene-grid wan-offloading"></span>

# Offloading

Рассмотрим генерацию следующего чанка. В прямоугольниках — необходимые данные и веса.

$$
\underbrace{\boxed{\mathrm{prompt}}}_{\text{промпт}}
\xrightarrow[\boxed{W^{\rm text}}]{\text{T5}}
\underbrace{\boxed{\mathrm{ctx}}}_{\text{текстовое условие}}
$$
$$
\left(\underbrace{\boxed{\epsilon_{n+1}}}_{\text{шум}},\boxed{\mathrm{ctx}},
\underbrace{\boxed{\mathrm{KVvid}_n}}_{\text{KV видео}}\right)
\xrightarrow[\boxed{W^{\rm DiT}}]{\text{DiT}}
\left(\underbrace{\boxed{\mathrm{KVvid}_{n+1}}}_{\text{обновлённый KV self-attention}},
\underbrace{\boxed{\mathrm{lat}_{n+1}}}_{\text{латентный чанк}}\right)
$$
$$
\boxed{\mathrm{lat}_{n+1}}
\xrightarrow[\boxed{W^{\rm dec}}]{\text{VAE-декодер}}
\underbrace{\boxed{\mathrm{frame}_{n+1}}}_{\text{кадры}}
\longrightarrow\mathrm{MP4}
$$
$$
\underbrace{\boxed{\mathrm{ctx}},\ \boxed{\mathrm{KVvid}_{n+1}}}_{\text{сохраняем между чанками}},\qquad
\underbrace{\boxed{\mathrm{KVtext}(\mathrm{ctx})},\ \boxed{\mathrm{lat}_{\rm win}},\ \boxed{\mathrm{state}_{\rm dec}}}_{\substack{\text{по режиму: KV текста, латенты для recache,}\\\text{состояние потокового декодера}}}.
$$

Объединим это с рабочими буферами и кодом вычислений в ресурсы.

$$
\begin{aligned}
\underbrace{\mathcal R}_{\text{ресурсы}}
={}&\underbrace{\mathcal R_W}_{\text{веса}}
\cup\underbrace{\mathcal R_{\rm state}}_{\text{KV, состояние декодера}}
\cup\underbrace{\mathcal R_x}_{\text{активации, рабочие буферы}}\\
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
\underbrace{\mathcal R_i\subseteq\mathcal R}_{\text{ресурсы узла }v_i}.
$$

| <span class="wan-offload-methods">Метод / источник</span> | Контекст → вычисление | Идея |
| :--- | :--- | :--- |
| **Граф времени жизни**<br>[ResourcePlan](../../src/wan_dit_lab/offloading/resource_schedule.py)<br>Родственные подходы: [SwapAdvisor](https://par.nsf.gov/servlets/purl/10191573), [TensorShunt](https://github.com/eladwf/TensorShunt/blob/main/python/tensorshunt/schedule.py) | $\displaystyle\begin{aligned}&\underbrace{\mathrm{first}(r)=\min\{i:r\in\mathcal R_i\}}_{\text{первое использование}}\\&\underbrace{\mathrm{last}(r)=\max\{i:r\in\mathcal R_i\}}_{\text{последнее использование}}\\&\mathrm{acquire}(r)\to v_{\mathrm{first}(r)}\to\cdots\to v_{\mathrm{last}(r)}\to\mathrm{release}(r)\end{aligned}$ | Когда ресурс нужен и когда его можно освободить. Перед использованием ждём загрузку, перед освобождением ждём завершение CUDA. |
| **Окна весов 2×2**<br>[BlockOffloader](../../src/wan_dit_lab/offloading/offload.py)<br>[Diffusers: group offloading](https://huggingface.co/docs/diffusers/optimization/memory#group-offloading) | $\displaystyle\underbrace{2\text{ блока в pinned RAM}}_{\text{подготовка}}\longrightarrow\underbrace{2\text{ блока в VRAM}}_{\text{вычисление}}$ | Исходная реализация для весов. Фиксированная глубина предзагрузки. |
| **Общий кэш ресурсов**<br>[ResourceCache](../../src/wan_dit_lab/offloading/resource_cache.py)<br>**Применяем** | $\displaystyle\begin{aligned}&\mathrm{VRAM}\rightleftarrows\mathrm{RAM}_{\rm cache}\rightleftarrows\mathrm{SSD}\\&\mathrm{active}(r)\Rightarrow\text{не вытесняем}\\&\mathrm{dirty}(r)\land\mathrm{evict}(r)\Rightarrow\mathrm{write}(r)\end{aligned}$ | Единые бюджеты. Сначала вытесняем ресурс с наиболее далёким следующим использованием; иначе по давности обращения. Чистый файл переиспользуем. |
| **Модель стоимости**<br>[FlexGen §4.2–4.3](https://arxiv.org/html/2303.06865#S4)<br>**Подключена к общему кэшу** | Сначала измеряем:<br>$\displaystyle\underbrace{\Delta t_{\rm prepare}(r),\ \Delta t_{\rm transfer}(r)}_{\text{подготовка и перенос ресурса}},\quad\underbrace{\Delta t_{\rm compute}(v_i)}_{\text{время выполнения узла}}$<br>Затем выбираем:<br>$\displaystyle\underbrace{\mathrm{plan}}_{\text{что, куда и когда переносить}},\qquad\min_{\mathrm{plan}}\widehat t_{\rm end}(\mathrm{plan})$<br>при бюджетах RAM / VRAM и зависимостях $G$. | Расписание по замерам. Наши профили: медианы; запас = максимум загрузки и минимум вычисления. Ранее подбирали только окна весов. |

$$
\Delta t_{\rm disk}(r)=
\frac{\underbrace{B_{\rm read}(r)}_{\text{прочитанные байты}}}{\underbrace{\beta_{\rm read}}_{\text{байт/с}}}
+
\frac{\underbrace{B_{\rm write}(r)}_{\text{записанные байты}}}{\underbrace{\beta_{\rm write}}_{\text{байт/с}}}.
$$

[Замеры и условия предыдущих прогонов](cards/H12.%20Подгрузка%20блоков%20даст%20полноразмерный%20BF16%20эталон.md#Предыдущие%20короткие%20прогоны)

**Источники:** [FlexGen §4.2–4.3](https://arxiv.org/html/2303.06865#S4) · [Diffusers: group offloading](https://huggingface.co/docs/diffusers/optimization/memory#group-offloading) · [PyTorch: mmap](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html#using-torch-load-mmap-true).
