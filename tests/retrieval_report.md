# Retrieval Test Report

- **Queries**: `14`
- **Requested settings**: `{'top_k': 3, 'search_type': 'similarity', 'mode': 'hybrid', 'semantic_weight': 0.5, 'bm25_weight': 0.5}`
- **Settings actually used**: `{'top_k': 3, 'search_type': 'similarity', 'mode': 'hybrid', 'semantic_weight': 0.5, 'bm25_weight': 0.5}`
- **Similarity Metric**: `Cosine Similarity (Dense)`
- **Generated**: `2026-09-12 01:11:19`

## Summary

- **Avg latency**: `2.083s`
- **Errors**: `0/14`
- **Ground-truth hit rate**: `8/11` (73%)

| Category | Hits | Total | Rate |
|---|---|---|---|
| مباشر (من جدول) | 1 | 1 | 100% |
| زمني (صلاحيات) | 0 | 1 | 0% |
| حالات خاصة (ماذا لو) | 1 | 1 | 100% |
| مركب (Decomposition) | 1 | 1 | 100% |
| غير مباشر (سيناريو) | 1 | 1 | 100% |
| ربط (Multi-Doc) | 0 | 1 | 0% |
| شروط (Conditional) | 1 | 1 | 100% |
| فني (SLA) | 0 | 1 | 0% |
| لغوي (Synonyms) | 1 | 1 | 100% |
| إجرائي (Process) | 1 | 1 | 100% |
| مباشر (Penalty) | 1 | 1 | 100% |

---

## Q1. السلام عليكم، ما هي جنسية اللاعب محمد صلاح؟

- **Category**: `خارج النطاق (OOD)`
- **Expected reference**: `اختبار رفض الإجابة من خارج النطاق.`
- **Latency**: `25.044s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Assets and wearhouse operation Tasks and Procedures Manual.pdf | 11 | markdown_heading | n/a | | نسؤول | | | |
| 2 | Central Mail and Files Unit Procedures Manual.pdf | 16 | markdown_heading | n/a | الحقيبة المخصصة له ) واقفالها محمد . وذلك تحكم بوسيلة | 6 | |
| 3 | Assets and wearhouse operation Tasks and Procedures Manual.pdf | 23 | markdown_heading | n/a | | | امنفذ | | |

---

## Q2. هاي، كيف حالك؟

- **Category**: `خارج النطاق (OOD)`
- **Expected reference**: `اختبار التفاعل الاجتماعي البسيط.`
- **Latency**: `0.287s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Central Mail and Files Unit Procedures Manual.pdf | 16 | markdown_heading | n/a | . | | |
| 2 | Central Mail and Files Unit Procedures Manual.pdf | 29 | markdown_heading | n/a | | ديوان اداري موظف | | | | الوحدة. حفظ نسخة من القرارفي الملف الخاص لدى | -11 | -11 | |
| 3 | Central Mail and Files Unit Procedures Manual.pdf | 16 | markdown_heading | n/a | | | |

---

## Q3. من هو صاحب صلاحية الموافقة على دخول جناح الإدارة في دليل الإنذار المركزي؟

- **Category**: `مباشر (من جدول)`
- **Expected reference**: `(Tabular Reasoning) صفحة 5، دليل الإنذار.`
- **Ground-truth check**: ✅ HIT (expected page(s): [5])
- **Latency**: `0.292s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Central Alarm Tasks And Procedures Manual.pdf | 5 | markdown_heading | n/a | | الانذار المركزي مساعد مدير | _الفرع مدير | شاملة السيرفر الفروع ) جميع الأبواب داخل الفرع | | الانذار المركزي ونائب رئيس - مدير الشؤون الادارية والا |
| 2 | Central Alarm Tasks And Procedures Manual.pdf | 5 | markdown_heading | n/a | | الموافقة المطلوب من جهة الشؤون الإدارية والهندسية | الموافقة المطلوبة من طرف الجهة الطالبة | بطاقة / الموقع المطلوب اصداربطاقة أونقل صلاحية اوتعريف  |
| 3 | Central Alarm Tasks And Procedures Manual.pdf | 1 | markdown_heading | n/a | دليل إجر اءات وحدة الانذار المركزي |

---

## Q4. متى يجب مراجعة لقطات الـ Logs الخاصة بمدراء الأنظمة الأمنية؟

- **Category**: `زمني (صلاحيات)`
- **Expected reference**: `(Temporal Reasoning) صفحة 17، دليل الإنذار.`
- **Ground-truth check**: ❌ MISS (expected page(s): [17])
- **Latency**: `0.309s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Central Alarm Tasks And Procedures Manual.pdf | 18 | markdown_heading | n/a | | 7. إجراءات مراجعة ال logsالخاصة بأعمال مدراء الأنظمة الأمنية | 7. إجراءات مراجعة ال logsالخاصة بأعمال مدراء الأنظمة الأمنية | 7. إجراءات مراجعة ال l |
| 2 | Central Alarm Tasks And Procedures Manual.pdf | 7 | markdown_heading | n/a | كل 3 بحد حسب الحاجة بفحص التسجيلات الانذار المركزي يتوجب عى -6 اقصى القيام وحدة _ اشهر . يتوجب عى البنك وبحيث يتم عمل الآتي : وحدة التاكد من الوضع الت |
| 3 | Central Alarm Tasks And Procedures Manual.pdf | 18 | markdown_heading | n/a | | في Parameters الأنظمة تغيير | | | | هذه الأنظمة في برمجة تغيير | | | |

---

## Q5. ماذا يحدث لو فقدت وحدة البريد مغلفاً مرسلاً من أحد مراكز العمل؟

- **Category**: `حالات خاصة (ماذا لو)`
- **Expected reference**: `(Edge Case) صفحة 29، دليل البريد.`
- **Ground-truth check**: ✅ HIT (expected page(s): [29])
- **Latency**: `0.424s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Central Mail and Files Unit Procedures Manual.pdf | 29 | markdown_heading | n/a | | اول - بريد مركزي وملفات اداري | امنفذ | امنفذ | | وتحويله لموظف من قبله الى ضياع اوتلف او في حال تعرض البريد الصادر استلام بريد الكتروني او مذكرة من |
| 2 | Assets and wearhouse operation Tasks and Procedures Manual.pdf | 16 | markdown_heading | n/a | | مسؤول أول - الموجودات وعمليات المستودعات وحدة | الجهة المنفذة امنفذ | | الإلكتروني والتحقق من استكمال كافة استلام طلبيات اللوازم والقرطاسية الموجودا |
| 3 | Central Mail and Files Unit Procedures Manual.pdf | 6 | markdown_heading | n/a | | بالنسبة لمراكزالعمل غيرالمطبق لديها الاتمتة يتم اتباع مايلي : الواردتم وضسعه داخل مغلفات استلام البريد بعد مطابقة محتوياته مع قائمة البريد نموذج قائ |

---

## Q6. كيف يتم التعامل مع حالات الجرد؟

- **Category**: `مبهم (طلب توضيح)`
- **Expected reference**: `اختبار كفاءة الـ Clarification وتجنب العشوائية.`
- **Latency**: `0.272s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Central Mail and Files Unit Procedures Manual.pdf | 22 | markdown_heading | n/a | | الإشراف على عملية والمستودعات ووضع التوصيات اللازمة. الجرد | -9 | | |
| 2 | Central Alarm Tasks And Procedures Manual.pdf | 5 | markdown_heading | n/a | | | رقم النسخة | |---------|------------------| | 08/2024 | تاريخ الإصدار | | 02/2026 | تاريخ آخر مراجعة | الانذار المركزي دليل إجراءات وحدة القواعد و |
| 3 | Assets and wearhouse operation Tasks and Procedures Manual.pdf | 8 | markdown_heading | n/a | | الجرد بعملية المعنيين الموظفين | | المنفذ | |

---

## Q7. ما هي إجراءات جرد الموجودات الثابتة سنوياً، ومن هم أعضاء اللجنة، وكيف تُعالج الفروقات؟

- **Category**: `مركب (Decomposition)`
- **Expected reference**: `يختبر تجميع معلومات من صفحات 5 و10 من دليل الموجودات.`
- **Ground-truth check**: ✅ HIT (expected page(s): [5, 10])
- **Latency**: `0.342s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Assets and wearhouse operation Tasks and Procedures Manual.pdf | 5 | markdown_heading | n/a | رئيسا مستودع موجودات / عضو وأمين السر امين أحد عضو مندوب عن دائرة أنظمة المعلومات / عضو مندوب مركز العمل / عضو وتكون مهام اللجنةكما يلي : وإجراء المعا |
| 2 | Assets and wearhouse operation Tasks and Procedures Manual.pdf | 23 | markdown_heading | n/a | | والقرطاسية والموجودات الثابتة 13. إجراءات صرف معاملات شراء اللوازم | والقرطاسية والموجودات الثابتة 13. إجراءات صرف معاملات شراء اللوازم | والقرطاسية |
| 3 | Assets and wearhouse operation Tasks and Procedures Manual.pdf | 5 | markdown_heading | n/a | السلامة والصحة المهنية. خطورتها وبالتنسيق مع حب دعات حفظ المواد يتوجب على -3 المستو وحدة درجة وحدة ومراكز العمل المختلفة احتياجات كافة -4 الجهات المعن |

---

## Q8. إذا تعطل نظام إنذار الحريق في أحد الفروع ولم يمكن إصلاحه في نفس اليوم، ما هو الإجراء الأمني البديل؟

- **Category**: `غير مباشر (سيناريو)`
- **Expected reference**: `استنتاج الحل من "ملاحظة" في صفحة 16، دليل الإنذار.`
- **Ground-truth check**: ✅ HIT (expected page(s): [16])
- **Latency**: `0.323s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Central Alarm Tasks And Procedures Manual.pdf | 16 | markdown_heading | n/a | 6. اجراءات تشغيل أنظمة انذارالحريق والسرقة والاتصال الآلي الأمن الانذارالمركزي مركز وحدة إداري أول - غرفة العمليات الجهة المنفذة المنفذ والسرقة والاتص |
| 2 | Central Alarm Tasks And Procedures Manual.pdf | 18 | markdown_heading | n/a | | إيقاف أحد الكواشف فيأحد المواقع | | | | إيقاف نظام الحريق اوأحدكواشف الحريق . | | | |
| 3 | Central Alarm Tasks And Procedures Manual.pdf | 7 | markdown_heading | n/a | كل 3 بحد حسب الحاجة بفحص التسجيلات الانذار المركزي يتوجب عى -6 اقصى القيام وحدة _ اشهر . يتوجب عى البنك وبحيث يتم عمل الآتي : وحدة التاكد من الوضع الت |

---

## Q9. ما هو دور نظام الـ BPM في تنظيم العلاقة بين وحدة البريد المركزي ووحدة الموجودات؟

- **Category**: `ربط (Multi-Doc)`
- **Expected reference**: `يربط بين دليل البريد (ص4) ودليل الموجودات (ص6).`
- **Ground-truth check**: ❌ MISS (expected page(s): [4, 6])
- **Latency**: `0.325s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Central Mail and Files Unit Procedures Manual.pdf | 16 | markdown_heading | n/a | يخص الدوائر المشمولة بالنظام يتم استلام البريد على ان يكون في مغلفات مغلقة ومرحل على نظام اتمتة عمليات تبادل البريد ترسل للفروع الائتمان والتي التحقق  |
| 2 | Central Mail and Files Unit Procedures Manual.pdf | 24 | markdown_heading | n/a | بعد ترحيلها على نظام اتمتة عمليات تبادل البريد BPM لاستكمال إجراءات السداد من قبلهم على نظام اتمتة عمليات تبادل البريد BPM لتدارهذه الحسابات من قبلهم  |
| 3 | Central Alarm Tasks And Procedures Manual.pdf | 7 | markdown_heading | n/a | وسلامة البنوك ومتطلبات أنظمة الضبط والرقابة الداخلية وحسب تعليمات البنك بمتطلبات الإنذار المركزي مراعاة يتوجب على -9 وتعديلاتها . والجهات ذات العلاقة  |

---

## Q10. متى يُسمح باستبدال بطاقة الدخول التالفة للموظف دون تغريمه مبلغ الـ 10 دنانير؟

- **Category**: `شروط (Conditional)`
- **Expected reference**: `اختبار فهم الاستثناءات، صفحة 5، دليل الإنذار.`
- **Ground-truth check**: ✅ HIT (expected page(s): [5])
- **Latency**: `0.308s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Central Alarm Tasks And Procedures Manual.pdf | 5 | markdown_heading | n/a | | | رقم النسخة | |---------|------------------| | 08/2024 | تاريخ الإصدار | | 02/2026 | تاريخ آخر مراجعة | الانذار المركزي دليل إجراءات وحدة القواعد و |
| 2 | Central Mail and Files Unit Procedures Manual.pdf | 9 | markdown_heading | n/a | | لوحدة الرواتب والنفقات العمليات المركزية. فيحال كانت المطالبة لغاية 500 ديناريتم تحويلها | لوحدة الرواتب والنفقات العمليات المركزية. فيحال كانت المط |
| 3 | Assets and wearhouse operation Tasks and Procedures Manual.pdf | 25 | markdown_heading | n/a | | لتنفيذ هذه الاعمال دون ملاحظات | لتنفيذ هذه الاعمال دون ملاحظات | -4 | |

---

## Q11. ما هي دورية مراجعة سجلات (Logs) نظام "Vanguard" ونظام "الإنذار المبكر"؟

- **Category**: `فني (SLA)`
- **Expected reference**: `اختبار استرجاع معلومات من الملاحق، ص 20، دليل الإنذار.`
- **Ground-truth check**: ❌ MISS (expected page(s): [20])
- **Latency**: `0.304s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Central Alarm Tasks And Procedures Manual.pdf | 7 | markdown_heading | n/a | كل 3 بحد حسب الحاجة بفحص التسجيلات الانذار المركزي يتوجب عى -6 اقصى القيام وحدة _ اشهر . يتوجب عى البنك وبحيث يتم عمل الآتي : وحدة التاكد من الوضع الت |
| 2 | Central Mail and Files Unit Procedures Manual.pdf | 28 | markdown_heading | n/a | | ديوان اداري موظف | المنفذ | المنفذ | | | | -2 | | سحلات ابداء ملفات تعبئة ملفات الاتلاف غير المباشر مستند | سحلات ابداء ملفات تعبئة ملفات الاتلاف غي |
| 3 | Central Alarm Tasks And Procedures Manual.pdf | 5 | markdown_heading | n/a | الإنذار المركزي استخراج تقرير للمخولين بالدخول على مراكز البيانات وغرف يتوجب على بشكل سنوي أوكلما تطلب وعكس اية تعديلات مطلوبة مراجعتها من الخوادم وال |

---

## Q12. ما هي شروط "إتلاف" الأغراض الموجودة في "المخازن"؟

- **Category**: `لغوي (Synonyms)`
- **Expected reference**: `اختبار مرونة اللغة والترادفات، صفحة 26، دليل الموجودات.`
- **Ground-truth check**: ✅ HIT (expected page(s): [26])
- **Latency**: `0.317s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Assets and wearhouse operation Tasks and Procedures Manual.pdf | 6 | markdown_heading | n/a | من المواد بعد اتلافها . السلامة والصحة المهنية بحالات الاتلاف من حيث آلية الاتلاف وآلية ملاحظة: يتوجب التنسيق مع التخلص وحدة -9 والزيادة في حالة ظهوره |
| 2 | Central Mail and Files Unit Procedures Manual.pdf | 9 | markdown_heading | n/a | | التأكد وتدقيق المطالبة الواردة ضمن البريد الالكتروني الشامل / على المذكرة التفصيلية والمعززات المرفقة واتخاذ القرار المناسب والتوصية وجود ما يعيق ذل |
| 3 | Assets and wearhouse operation Tasks and Procedures Manual.pdf | 26 | markdown_heading | n/a | | المطبعة والمستودعات مدير | | | بالإتلاف وحسب الصلاحيات المعتمدة بالخصو التنسيب ص . | -8 | | الموجودات وعمليات المستودعات الموظف _والسير بإجراءات الا |

---

## Q13. كيف يتم استخراج لقطات كاميرات المراقبة لجهة خارجية مثل "البحث الجنائي"؟

- **Category**: `إجرائي (Process)`
- **Expected reference**: `تلخيص إجراءات صفحة 11، دليل الإنذار.`
- **Ground-truth check**: ✅ HIT (expected page(s): [11])
- **Latency**: `0.344s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Central Alarm Tasks And Procedures Manual.pdf | 11 | markdown_heading | n/a | | 3. إجراءات استخراج اللقطات المسجلة من انظمة المراقبة التلفزيونية | 3. إجراءات استخراج اللقطات المسجلة من انظمة المراقبة التلفزيونية | |------------- |
| 2 | Central Alarm Tasks And Procedures Manual.pdf | 7 | markdown_heading | n/a | فتح كاميرات الفروع التي تتأخر التاكد منان فترة التسجيل لانظمة المراقبة التلقزيونية مطابقة لتعليمات البنك المركزي ووزارة الداخلية بشكل شهري. توثيق انوا |
| 3 | Assets and wearhouse operation Tasks and Procedures Manual.pdf | 19 | markdown_heading | n/a | | البنك مستودعات والقرطاسية الموردة من جهات خارجية اجراءات استلام 10 -لدى اللوازم | البنك مستودعات والقرطاسية الموردة من جهات خارجية اجراءات استلام 10 |

---

## Q14. كم تبلغ غرامة فقدان بطاقة الدخول؟

- **Category**: `مباشر (Penalty)`
- **Expected reference**: `صفحة 5، دليل الإنذار.`
- **Ground-truth check**: ✅ HIT (expected page(s): [5])
- **Latency**: `0.279s`
- **Chunks retrieved**: `3`

| Rank | Source | Page | Strategy | Score | Preview |
|---|---|---|---|---|---|
| 1 | Central Mail and Files Unit Procedures Manual.pdf | 26 | markdown_heading | n/a | | | -10 | |
| 2 | Central Alarm Tasks And Procedures Manual.pdf | 5 | markdown_heading | n/a | | | رقم النسخة | |---------|------------------| | 08/2024 | تاريخ الإصدار | | 02/2026 | تاريخ آخر مراجعة | الانذار المركزي دليل إجراءات وحدة القواعد و |
| 3 | Central Mail and Files Unit Procedures Manual.pdf | 29 | markdown_heading | n/a | | مدير - المطبعة والمستودعات | المنفذ | المنفذ | | من او ما يفيد عدم موافقة الشركة على قيمة اداري التغريم الخصم | -8 | -8 | |

---
