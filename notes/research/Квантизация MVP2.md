<span class="wan-scene-style wan-scene-math wan-scene-grid"></span>

# Квантизация · MVP2

Что меняется в квантователе при спуске от эталона $704\times1280$? Рассмотрим квантователь MVP1 как один фактор конфигурации.

$$
\underbrace{\mathcal Q^{(1)}=(\mathrm{W4A4},\ \mathrm{INT8},\ \mathrm{FP8\ E4M3})}_{(W_{\rm DiT},\ W_{\rm T5},\ \mathrm{KVvid})\text{ — [[Квантизация|MVP1]]}}
\qquad\longmapsto\qquad
\underbrace{\mathcal Q^{(2)}=\mathcal Q^{(1)}\ \text{кроме}\ W_{\rm T5}:\ \mathrm{INT8}\mapsto\mathrm{BF16}}_{\text{MVP2}}
$$

Теперь рассмотрим, насколько далеко друг от друга два ролика одной конфигурации и одного seed: ниже этого расстояния потеря не различает квантователи.

$$
\underbrace{\mathcal L_{\rm floor}=\mathcal L\bigl(\mathrm{video}(\mathrm{cfg},s),\ \mathrm{video}'(\mathrm{cfg},s)\bigr)}_{\text{повторный прогон или возмущение входа на }10^{-5}},\qquad
\underbrace{\mathcal L_{\rm ceil}=0.736}_{\text{другой seed}}
$$

| Преобразование | Контекст → вычисление | Идея | Наблюдения |
| :--- | :--- | :--- | :--- |
| <span class="wan-kept">Применяем</span><br>$\displaystyle W_{\rm DiT}:\ \mathrm{W4A4}\xrightarrow{\rm id}\mathrm{W4A4}$<br>проверено против FP8 на эталоне | $\displaystyle \begin{aligned}&\text{Рассмотрим чекпойнт LongLive-2.0-5B-NVFP4-S2.}\\[5pt]&\underbrace{\theta=\arg\min\ \mathcal L_{\rm train}\bigl(\Phi(\cdot\,;\mathcal Q_{\rm NVFP4}(W))\bigr)}_{\text{обучен и дистиллирован с 4-битными весами и активациями}}\\[5pt]&\text{Поэтому эталонный путь этого чекпойнта 4-битный, а не FP32.}\end{aligned}$ | Для этого чекпойнта ближе к FP32 не значит лучше видео. Не проверено: NVFP4 с блочными масштабами при обучении против нашего W4A4 SVDQuant. | $\displaystyle \begin{aligned}&\underbrace{\mathrm{FPS}:2.17\to5.41}_{\mathrm{FP8}\to\mathrm{W4A4},\ 704\times1280}\\[5pt]&\text{В просмотре (сетка 03.11): W4A4 не хуже FP8.}\end{aligned}$<br>[E30](cards/E30.%20%D0%A1%D1%82%D0%B0%D1%80%D1%82%D0%BE%D0%B2%D0%B0%D1%8F%20%D1%82%D0%BE%D1%87%D0%BA%D0%B0%20W4A4%20704%C3%971280%20%D0%B8%20%D0%BE%D1%82%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5%20%D1%88%D0%B0%D0%B3%D0%B8.md) |
| <span class="wan-kept">Применяем</span><br>$\displaystyle W_{\rm T5}:\ \mathrm{INT8}\mapsto\mathrm{BF16}$ | $\displaystyle \begin{aligned}&\text{Рассмотрим векторы промпта }\mathrm{ctx}=\mathrm{T5}(\mathrm{prompt};W_{\rm T5}).\\[5pt]&\text{Было: }\widehat W_{\rm T5}^{\rm INT8}.\quad\text{Стало: }W_{\rm T5}^{\rm BF16}.\end{aligned}$ | Возвращаем энкодер в 16 бит и платим задержкой промпта. INT8 отложен на конец спуска. | $\displaystyle \begin{aligned}&\text{В просмотре: с INT8 видео хуже следует промпту.}\\[5pt]&T_{\rm prompt}:1.63\to2.86\ \mathrm{s}\\[5pt]&\underbrace{\mathrm{FPS}:5.58\to5.41}_{704\times1280}\end{aligned}$<br>[E30](cards/E30.%20%D0%A1%D1%82%D0%B0%D1%80%D1%82%D0%BE%D0%B2%D0%B0%D1%8F%20%D1%82%D0%BE%D1%87%D0%BA%D0%B0%20W4A4%20704%C3%971280%20%D0%B8%20%D0%BE%D1%82%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5%20%D1%88%D0%B0%D0%B3%D0%B8.md) |
| <span class="wan-kept">Применяем</span> в экспериментах<br>$\displaystyle \textstyle\sum_{\rm parallel}\mapsto\sum_{\rm ordered}$<br>`kernels.lora_sum` | $\displaystyle \begin{aligned}&\text{Рассмотрим выход слоя W4A4 как сумму слагаемых }a_{r}.\\[5pt]&\text{Было: }y=\textstyle\sum_r a_r\ \text{складывают параллельные потоки, порядок не задан.}\\[5pt]&\text{Стало: те же слагаемые складываются в фиксированном порядке.}\end{aligned}$ | Сложение float не ассоциативно, поэтому порядок меняет последние биты, и ролик расходится. Фиксированный порядок делает повтор побитно равным, Live оставляет параллельную сумму (`atomic`). | $\displaystyle \begin{aligned}&\underbrace{\mathcal L_{\rm floor}:0.26\,/\,0.18\to0}_{\text{повторный прогон, лиса / чашка}}\\[5pt]&\underbrace{-2\%\ \mathrm{FPS}}_{\text{ordered}},\qquad\underbrace{+1.7\%\ \mathrm{FPS}}_{\text{atomic}}\end{aligned}$<br>[Отчёт, §31](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |
| Замер<br>$\displaystyle \mathcal L_{\rm floor}(\mathcal Q)$ | $\displaystyle \begin{aligned}&\text{Рассмотрим возмущение входа на }10^{-5}\text{ при }704\times1280.\\[5pt]&\mathcal L_{\rm floor}=\mathcal L\bigl(\mathrm{video}(x),\mathrm{video}(x+10^{-5}\xi)\bigr)\end{aligned}$ | Расхождение близких входов это свойство модели, 4-битный путь его удваивает. Парная потеря ниже пола результатом не считается. | $\displaystyle \begin{aligned}&\underbrace{0.12}_{\mathrm{FP8}+\mathrm{SDPA}}\to\underbrace{0.18}_{\mathrm{FP8}+\mathrm{Sage}}\to\underbrace{0.33}_{\mathrm{W4A4}+\mathrm{Sage}}\end{aligned}$<br>[Отчёт, часть 3](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |
| Не применяем: разрыв не из-за калибровки<br>$\displaystyle s_{704}\mapsto s_{608}$ | $\displaystyle \begin{aligned}&\text{Рассмотрим сглаживание }s\text{ W4A4, откалиброванное при }704\times1280.\\[5pt]&\text{Сравниваем потерю при }608\text{ с точными весами FP8 и с W4A4.}\end{aligned}$ | Если бы W4A4 проигрывал из-за чужой калибровки, FP8 был бы ближе к эталону. Это не так, поэтому одна калибровка служит всем разрешениям. | $\displaystyle \begin{aligned}&\underbrace{\overline{\mathcal L}_1:0.382\ \text{против}\ 0.373}_{\mathrm{FP8}\ \text{против}\ \mathrm{W4A4},\ 608}\end{aligned}$<br>[Отчёт, §31](../../docs/report-2026-11-11-loss-704-608-for-fable.md) |

$$
\mathcal Q^{(2)}=\bigl(\underbrace{\mathrm{W4A4}}_{W_{\rm DiT}},\ \underbrace{\mathrm{BF16}}_{W_{\rm T5}},\ \underbrace{\mathrm{FP8\ E4M3}}_{\mathrm{KVvid}}\bigr)
$$

[Конфиг рабочей точки](../../configs/candidates/544x960-win32-live.json) · Эксперимент: [E30](cards/E30.%20%D0%A1%D1%82%D0%B0%D1%80%D1%82%D0%BE%D0%B2%D0%B0%D1%8F%20%D1%82%D0%BE%D1%87%D0%BA%D0%B0%20W4A4%20704%C3%971280%20%D0%B8%20%D0%BE%D1%82%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5%20%D1%88%D0%B0%D0%B3%D0%B8.md)
