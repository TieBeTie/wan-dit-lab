<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

# Позиции RoPE на кадре меньше обученного

Как задать позиции RoPE для высоты и ширины, когда кадр меньше обученного (512×928 против 704×1280), чтобы
кадр оставался связным и детальным?

$$
\underbrace{s=\frac{L'}{L}}_{\text{новая длина к обученной}},\qquad
\underbrace{s_{\rm H}=\frac{512/32}{704/32}=\frac{16}{22}\approx0.73}_{\text{у нас осталось 16 из 22 токенов по высоте}},\qquad
\underbrace{\theta_d=b^{-2d/\lvert D\rvert}}_{\text{частоты RoPE}}
$$

Все три статьи ниже решают обратную задачу, $s>1$: расширение длины. Ни одна не рассматривает $s<1$; PI и
YaRN берут только $L'>L$, динамический YaRN ограничивает $s=\max(1,l'/L)$, RIFLEx считает малое расширение
решённым без изменений. Перенос на $s<1$ ниже — наша гипотеза.

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| $\displaystyle m\mapsto\frac{mL}{L'}$<br>[Position Interpolation, §2.3, (4)](https://arxiv.org/html/2306.15595) | $\displaystyle \begin{aligned}&\mathbf f'(\mathbf x,m)=\mathbf f\!\left(\mathbf x,\frac{mL}{L'}\right)\\[6pt]&\underbrace{\frac{L}{L'}=\frac{1}{s}}_{\text{один множитель на все частоты}}\\[8pt]&\text{у нас }s=0.73:\ \underbrace{m\mapsto1.375\,m}_{\texttt{rope.stretch}=1}\\[6pt]&\texttt{rope.stretch}=t:\quad m\mapsto(1/s)^t\,m\end{aligned}$ | Позиции переводятся в обученный диапазон вместо экстраполяции; по §2.3 интерполяция «much more stable and will not lead to wild values». Для $s<1$ это растяжение позиций до сетки 704×1280. | Применяем (`rope.stretch`). Космонавт, W4A4, 512×928, масштаб позиций: ×0.73, ×0.85, ×0.9, ×1.0, ×1.05, ×1.1, ×1.15, ×1.17, ×1.2, ×1.375, ×1.49. Пользователь: лучше ×1.05–×1.1, хуже полная PI ×1.375 и сжатие ниже ×1. Видео: `quality-search/astronaut-scales-2026-11-04.mp4`, `grid-stretch-*.mp4`. |
| $\displaystyle b\mapsto b'=b\,s^{\lvert D\rvert/(\lvert D\rvert-2)}$<br>[YaRN, App. A.2, (17)–(20)](https://arxiv.org/html/2309.00071) («NTK-aware») | $\displaystyle \begin{aligned}&g(m)=m,\quad h(\theta_d)=b'^{-2d/\lvert D\rvert}\\[6pt]&\underbrace{b'=b\,s^{\lvert D\rvert/(\lvert D\rvert-2)}}_{\text{меняем основание}}\\[8pt]&\underbrace{d=0:\ \theta_0\ \text{не меняется}}_{\text{высокая частота}},\quad\underbrace{d\to\lvert D\rvert/2:\ \theta_d\to\theta_d/s}_{\text{низкая частота}}\\[10pt]&\underline{\text{у нас}}\\[4pt]&b'_{\rm H}\approx7158\\[2pt]&b'_{\rm W}\approx7134\end{aligned}$ | Высокие частоты почти не трогаем, низкие интерполируем как PI. В A.2 отмечено, что на практике $s$ приходится брать больше теоретического. | Не применяем отдельно: гладкий частный случай масштаба по частотам; проверяем by-parts. |
| $\displaystyle \theta_d\mapsto h(\theta_d)$<br>[YaRN, §3.2, (8), (10)–(13)](https://arxiv.org/html/2309.00071) («NTK-by-parts») | $\displaystyle \begin{aligned}&\underbrace{\lambda_d=\frac{2\pi}{\theta_d}}_{\text{длина волны}},\quad \underbrace{r(d)=\frac{L}{\lambda_d}}_{\text{сколько волн на длине}}\\[8pt]&\gamma(r)=\begin{cases}0,&r<\alpha\\1,&r>\beta\\\dfrac{r-\alpha}{\beta-\alpha},&\text{иначе}\end{cases}\\[8pt]&h(\theta_d)=\bigl(1-\gamma(r(d))\bigr)\frac{\theta_d}{s}+\gamma(r(d))\,\theta_d\\[6pt]&\text{для Llama: }\alpha=1,\ \beta=32\\[10pt]&\underline{\text{у нас, высота}}\\[4pt]&r(d)=3.5,\,2.3,\,1.5,\,0.9,\dots\\[4pt]&\alpha{=}1,\beta{=}32:\\[2pt]&\ \ 1.35,\,1.36,\,1.37,\,1.375\dots\\[4pt]&\alpha{=}0.25,\beta{=}2:\\[2pt]&\ \ 1.0,\,1.0,\,1.12,\,1.23,\,1.30\dots\end{aligned}$ | Частоты, у которых на длине $L$ много волн ($r>\beta$), не трогаем; частоты с меньше чем одной волной ($r<\alpha$) интерполируем как PI; между ними линейный переход. | Применяем (`rope.by_parts`), оба порога, космонавт 10.9 / 10.8 fps. Видео: `quality-search/astronaut-by-parts-2026-11-04.mp4` (старт, ×1.1, α=1 β=32, α=0.25 β=2). Оценка пользователя ещё не дана. |
| $\displaystyle \operatorname{softmax}\!\left(\frac{q^\top k}{t\sqrt{\lvert D\rvert}}\right)$<br>[YaRN, §3.3, (14)–(15)](https://arxiv.org/html/2309.00071) | $\displaystyle \begin{aligned}&\sqrt{\frac{1}{t}}=0.1\ln s+1\\[6pt]&\text{подобрано на LLaMA при }s>1\\[10pt]&\underline{\text{у нас}}\\[4pt]&\sqrt{1/t}=0.968\\[2pt]&1/t=0.937\end{aligned}$ | Поправка температуры attention вместе с NTK-by-parts. | Не применяем: формула подобрана при $s>1$, при $s<1$ вне области подбора; в ядро Sage её пришлось бы встраивать отдельно. |
| $\displaystyle \theta_k\mapsto\frac{2\pi}{Ls}$<br>[RIFLEx, §3.2–3.3, (4)–(8)](https://arxiv.org/html/2502.15894) | $\displaystyle \begin{aligned}&\underbrace{N_j=\frac{2\pi}{\theta_j}}_{\text{период}},\quad \underbrace{k=\arg\min_j\lvert N_j-N\rvert}_{\text{частота повтора сюжета}}\\[8pt]&\underbrace{N'_k\ge Ls\ \Rightarrow\ \theta'_k\le\frac{2\pi}{Ls}}_{\text{один период на всю длину}}\\[6pt]&\text{остальные }\theta_j\text{ без изменений}\\[10pt]&\underline{\text{у нас}}\\[4pt]&N_j=6.3,\,9.7,\,15,\,23,\dots\\[2pt]&\text{повтора нет}\end{aligned}$ | Меняется одна «внутренняя» частота, у которой период близок к месту первого повтора; применяется и к высоте с шириной (§4.2, App. D). | Не применяем: лечит повтор сюжета при удлинении, у нас этого дефекта нет. |

Сейчас сравниваем PI одним числом (лучшее ×1.05–×1.1) и масштаб по частотам с двумя порогами.

Дата записи: 04.11.2026.
