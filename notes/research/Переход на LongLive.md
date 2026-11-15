# Переход на LongLive · MVP1

Что изменили после экспериментов с Wan?

$$
\underbrace{\mathrm{Wan/Self\mbox{-}Forcing}}_{\text{свои методы работы с KV}}
\longrightarrow
\underbrace{\mathrm{LongLive\mbox{-}1.3B}}_{\text{сменили веса}}
\longrightarrow
\underbrace{\mathrm{LongLive\mbox{-}2.0\mbox{-}5B\ S2}}_{\text{выбрали для MVP1}}.
$$

| Изменение | Наблюдение / результат | Эксперименты |
| :--- | :--- | :--- |
| $\displaystyle W_{\rm SF}\mapsto W_{\rm LL\,1.3B}$ | $\displaystyle \begin{aligned}&480\times832,\quad\mathrm{FPS}=1.75\text{–}1.93\\&\text{В сравнении E11 пользователь выбрал LongLive.}\\&\underbrace{60\ \mathrm{s}}_{\text{серия E12 без отмеченного дрейфа}}\\&\text{На третьем промпте появлялся второй кот.}\end{aligned}$ | [E11](cards/E11.%20LongLive-1.3B%20%D1%81%D0%BC%D0%B5%D0%BD%D0%B0%20%D1%81%D1%86%D0%B5%D0%BD%D1%8B,%20480p%2030%20%D1%81.md) · [H14](cards/H14.%20%D0%9F%D0%BE%D0%BB%D0%BD%D1%8B%D0%B9%20LongLive%20%D0%BB%D1%83%D1%87%D1%88%D0%B5%20%D1%81%D0%BE%D0%BE%D1%82%D0%B2%D0%B5%D1%82%D1%81%D1%82%D0%B2%D1%83%D0%B5%D1%82%20%D0%B4%D0%B5%D0%BC%D0%BE%D0%BD%D1%81%D1%82%D1%80%D0%B0%D1%86%D0%B8%D0%B8.md) · [E12](cards/E12.%20LongLive-1.3B%20%D0%BF%D0%BE%D0%BA%D0%B0%D0%B7%D0%B0%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5%20%D0%BF%D1%80%D0%BE%D0%BC%D0%BF%D1%82%D1%8B,%2060%20%D0%B8%2030%20%D1%81.md) |
| $\displaystyle W_{\rm LL\,1.3B}\mapsto W_{\rm LL\,2.0\,5B}$ | $\displaystyle \begin{aligned}&\text{E14: действия кота лучше, второго кота нет;}\\&\text{смена сцен хуже, чем у 1.3B.}\\&\underbrace{\mathrm{multi\mbox{-}shot\ sink}+\mathrm{narrative\ RoPE}}_{\text{E15: перенесли из LongLive 2}}\\&\text{Чистый пляж; возврат в лес.}\\&\mathrm{FPS}=2.9,\quad T_{\rm switch}=1.7\text{–}2.5\ \mathrm{s}\end{aligned}$ | [E14](cards/E14.%20LongLive%202%20%D0%BF%D1%80%D0%BE%D1%82%D0%B8%D0%B2%201.3B%20%D0%BD%D0%B0%20%D0%BF%D0%BE%D0%BA%D0%B0%D0%B7%D0%B0%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D1%8B%D1%85%20%D0%BF%D1%80%D0%BE%D0%BC%D0%BF%D1%82%D0%B0%D1%85.md) · [E15](cards/E15.%20LongLive%202%20%D1%81%D0%BA%D0%BB%D0%B5%D0%B9%D0%BA%D0%B8%20%D1%81%20multi-shot%20sink.md) |
| $\displaystyle \begin{aligned}&\mathrm{LongLive\,2\ S2}\\&+\mathrm{W4A4}+\mathrm{Sage\,2}\end{aligned}$ | $\displaystyle \begin{aligned}&480\times832,\quad\text{полный KV-кэш}\\&\underbrace{\mathrm{FPS}:4.6\to16.4}_{\text{изменены шаги, точность и исполнение}}\\&M_{\rm GPU}^{\rm peak}=5.83\ \mathrm{GiB}\\&\underbrace{6\ \text{чанков}}_{\text{в просмотре сцена связная}}\end{aligned}$ | [E26](cards/E26.%20Live%20W4A4%20%D0%B8%20Sage%202,%20480%C3%97832.md) |

**Итог: продолжение видео в MVP1**

$$
\begin{aligned}
&\underbrace{\mathrm{DiT}=\mathrm{LongLive\mbox{-}2.0\mbox{-}5B\ S2}}_{\text{вместо прежнего Wan/Self-Forcing}}\\
&\underbrace{(F_{\rm chunk},F_{\rm anch},F_{\rm win})=(8,8,32)}_{\text{чанк, якорь, окно}}\\
&(\mathrm{KVvid}_n,\mathrm{ctx},\epsilon_{n+1})
\xrightarrow{\Phi_{\rm continue}^{(1)}}
(\mathrm{lat}_{n+1},\mathrm{KVvid}_{n+1}),\\
&\underbrace{N_{\rm steps}=2,\quad\mathrm{shift}=5}_{\text{рабочая конфигурация}}.
\end{aligned}
$$

[Конфигурация MVP1](../../configs/live.json) · [Сводка версий](../../Версии%20исследования.canvas)
