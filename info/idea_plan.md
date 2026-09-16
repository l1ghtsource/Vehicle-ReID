Старое (2021): [https://github.com/damo-cv/TransReID](https://github.com/damo-cv/TransReID) (Transformer-based Object Re-Identification) \- просто для исторической справки.

В 2026 вышло [Rethinking Multi-Branch and Cross-Backbone Fusion for Vehicle Re-Identification in the Foundation-Model Era](https://arxiv.org/abs/2607.22068) \- тут говорят что просто сам dinov3-convnext с небольшим тюном трахает все на свете и все эти замудренные старые пайпы с ReID не нужны. при этом vit dino у них хуже.

Наверное много накопать можно на [https://huggingface.co/papers?q=Object+re-identification](https://huggingface.co/papers?q=Object+re-identification) 

1. Бекбоны:  
1) [https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m)  
2) [https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m)  
3) [https://huggingface.co/nvidia/C-RADIOv4-SO400M](https://huggingface.co/nvidia/C-RADIOv4-SO400M) (дистилл сразу из 7b dino \+ siglip2 \+ sam3 от нвидиа)  
4) [https://huggingface.co/microsoft/LLM2CLIP-EVA02-L-14-336](https://huggingface.co/microsoft/LLM2CLIP-EVA02-L-14-336) (для разнообразия, eva02 вроде топ даже щас)  
5) [https://huggingface.co/occurra/vehicle\_vit\_clip\_reid](https://huggingface.co/occurra/vehicle_vit_clip_reid) (тюн [Syliz517/CLIP-ReID](https://github.com/Syliz517/CLIP-ReID)  напрямую под vehicle reid)

2. Мб перед файнтюном на нашей дате дополнительно потренить на следующих датасетах:  
1) банально тест+трейн данные, но без лейблов \- dino-line ssl pretraining  
2) [huggingface.co/datasets/yandex/mad-cars](http://huggingface.co/datasets/yandex/mad-cars) \- супервайзд трейн на клф марки+ориентации. или использовать на других стадиях для псевдолейбла, публичный  
3) [https://github.com/PKU-IMRE/VERI-Wild](https://github.com/PKU-IMRE/VERI-Wild) ([https://hyper.ai/en/datasets/9396](https://hyper.ai/en/datasets/9396)) \- буквально тематический датасет 1 в 1 как в сореве (+старенькая статья: [https://openaccess.thecvf.com/content\_CVPR\_2019/papers/Lou\_VERI-Wild\_A\_Large\_Dataset\_and\_a\_New\_Method\_for\_Vehicle\_CVPR\_2019\_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lou_VERI-Wild_A_Large_Dataset_and_a_New_Method_for_Vehicle_CVPR_2019_paper.pdf) \- там учили на триплет лоссе), не оч понял публичный ли он  
4) [https://github.com/JDAI-CV/VeRidataset](https://github.com/JDAI-CV/VeRidataset), публичный  
5) [~~https://pkuml.org/resources/pku-vehicleid.html~~](https://pkuml.org/resources/pku-vehicleid.html) ~~(приватный, надо делать запрос на почту. оказалась лицензия не ок)~~  
6) [https://www.aicitychallenge.org/2020-data-and-evaluation/](https://www.aicitychallenge.org/2020-data-and-evaluation/) (соревка, был кейс City-Scale Multi-Camera Vehicle Re-Identification \- есть данные), публичный  
7) [https://qmul-vric.github.io/](https://qmul-vric.github.io/), публичный

3. Тюн на дате соревы:  
1) ArcFace/SphereFace2 \+ P/K batch sampling  
2) Triplet Loss w/ mining  
3) AdaSP loss: [github.com/Astaxanthin/AdaSP](http://github.com/Astaxanthin/AdaSP)  
4) RPTM loss: [https://github.com/adhirajghosh/RPTM\_reid](https://github.com/adhirajghosh/RPTM_reid)   
5) комбинировать лоссы  
6) Augs: все базовое тут, особенно важно разные ракурсы, освещения, шакализации и тд (hflip, resized crop, colorjitter, gamma, explosure, jpeg, gaussian blur, motion blur, noise, fog/rain/snow augs, random erasing, coarse dropout)  
7) Val: GKF, или просто group-disjoint между трейн-вал, поскольку в реальном тесте ожидаются новые айдишники  
8) Input: ббокс кроп / ббокс кроп \+ внешний контекст  
9) сначала мб фризить бекбон первые k% степов  
10)  pooling: gap/gem/attn, multi-level (важно для конвнекста), bnneck (важно для поисковых штук)

4. Трюки:  
1) TTA на инфере (отражения, повороты, ..) \- оч важно  
2) Regularization: R-Drop, DropPath (особенно при псевдолейбл трейне), AWP, EMA  
3) Псевдолейблинг: если учим на триплетах, то можно просто намайнить новые тройки. А если ArcFace-like, то хорошо работает подход с HDBSCAN-кластеризацией и итеративным псевдолейблингом.  
4) гпт насрал: “global embedding даёт top-K кандидатов, затем top-K перепроверяются локальными correspondences, например DINO dense patches / local descriptors \+ mutual matching, LoFTR/LightGlue-подобная проверка”  
5) для refusal обучить катбуст или типа того (по ТЗ система должна уметь вернуть пустой ответ, если query-машины в галерее нет, вместо того чтобы всегда выдавать ближайшую). можно конечно просто потюнить пороги на OOF, но можно обучить маленький бинклф (трейн собрать легко, специально из результатов выдачи убрать машину с нужным id \- для oof мы знаем все id. так собрать себе 50/50 balanced 0/1 трейн выборку вида (ретривленный набор картинок, запрос) \-\> {есть ли среди набора картинок машина которую запросили}). фичи всякие статистики по сходствам внутри этого набора \+ tta разногласия (или разногласия разных ансамбль моделей) \+ графовые метрики

5. Постпроцессинг:  
1) k-reciprocal re-ranking ([https://openaccess.thecvf.com/content\_cvpr\_2017/papers/Zhong\_Re-Ranking\_Person\_Re-Identification\_CVPR\_2017\_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf)) \- оч важно  
2) aqe / query expansion ([https://journals.sagepub.com/doi/10.1177/15501477211066305](https://journals.sagepub.com/doi/10.1177/15501477211066305)) \- хз  
3) gallery aggregation \- emb \= a \* emb \+ (1-a) \* prototype\_emb  
4) S-Norm, AS-Norm  
5) graph-based reranking 

6. Идеи для презы:  
1) разобрать статьи и подходы  
2) показать экстра датку и что она нам дала  
1) сравнить разные бекбоны  
2) сравнить триплет/arcface подходы  
3) потестить те специальные лоссы  
4) показать постпроц \+ тта  
5) модельку в tensor-rt или прости господи onnx сбурмалдить  
6) пост-хок анализ: сделать для инпут картинки k зашумленных/искаженных версий и сравнить результаты ретрива с оригинальной картинкой и искаженной \- насколько изменились соседи, эмбеды и тд. это покажет устойчивость нашего эмбеддера ко всякому говну и искажениям