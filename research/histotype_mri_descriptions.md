# MRI appearance descriptions for five soft-tissue tumor histotype groups (prompt-ready, sourced)

Prepared for: MSTT-199 histotype categories F / L / M / N / V, to be pasted verbatim into a
vision-language-model prompt that localizes the tumor on a single MRI slice.

**Sequence conventions used below.** "T1" = non-fat-suppressed T1-weighted; "T2 FS" =
T2-weighted or other fluid-sensitive sequence with fat suppression (T2 fat-sat / STIR).
Signal is described relative to skeletal muscle and subcutaneous fat, as radiologists do.

---

## 0. What MSTT-199 actually contains (checked against the primary source)

Direct evidence (primary source, fetched): the MSTT-199 authors queried their orthopaedic
oncology registry for patients with a final pathologic diagnosis of: **schwannoma (benign
nerve tumor), MPNST (malignant peripheral nerve sheath tumor), well-differentiated
liposarcoma (benign fat tumor), dedifferentiated liposarcoma (malignant fat tumor), desmoid
fibromatosis (benign fibrous tumor), undifferentiated pleomorphic sarcoma (malignant fibrous
tumor), hemangioma or arteriovenous malformation (benign vascular tumor), angiosarcoma
(malignant vascular tumor), myxoma (benign myxoid tumor), or myxoid fibrosarcoma (malignant
myxoid tumor)**. These were divided into five categories: Fibrous, Fat, Myxoid, Nerve,
Vascular (≈40 patients each; 199 total). Tumors had to be > 3 cm and have both an axial T1
and an axial T2 fat-saturation sequence. Annotations were made **primarily on T2 images,
except Fat tumors, which were mostly annotated on T1 due to better tumor visibility**.
Source: Reasat T, Chenard S, Rekulapelli A, Chadwick N, Shechtel J, van Schaik K, Smith DS,
Lawrenz J. *MSTT-199: MRI Dataset for Musculoskeletal Soft Tissue Tumor Segmentation.*
arXiv:2409.03110 [eess.IV], 4 Sep 2024. https://arxiv.org/abs/2409.03110 (abstract page and
full HTML v1 both fetched and read).

Implication (researcher inference, flagged): the dataset's category membership is **narrower**
than the example entities listed in the task prompt. Notably:
- **F** in MSTT-199 = desmoid fibromatosis + undifferentiated pleomorphic sarcoma. It does
  **not** contain solitary fibrous tumor or fibrosarcoma.
- **L** in MSTT-199 = well-differentiated liposarcoma + dedifferentiated liposarcoma (pure
  lipoma is **not** listed, although the task groups lipoma with L).
- **M** in MSTT-199 = myxoma + myxoid fibrosarcoma (myxofibrosarcoma and myxoid liposarcoma
  are **not** explicitly listed, though "myxoid fibrosarcoma" is the closest entry and
  myxoid liposarcoma is a fat/myxoid hybrid).
- **N** in MSTT-199 = schwannoma + MPNST (neurofibroma is **not** listed).
- **V** in MSTT-199 = hemangioma/AVM + angiosarcoma.

The short descriptions below were written to cover the categories as the task groups them
(desmoid/SFT/fibrosarcoma for F; lipoma + ALT/WDLS for L; myxoma/myxoid liposarcoma/
myxofibrosarcoma for M; schwannoma/neurofibroma/MPNST for N; hemangioma/venous
malformation/angiosarcoma for V), because that is what the prompt consumer asked for. Notes
on where a description may under-cover a dataset entity (e.g. dedifferentiated liposarcoma,
angiosarcoma) are in section 3.

---

## 1. Prompt-ready short descriptions (each ≤ 60 words)

### F — Fibrous / fibroblastic–myofibroblastic (desmoid-type fibromatosis, solitary fibrous tumor, fibrosarcoma)

> Fibrous tumor: a soft-tissue mass isointense to slightly darker than adjacent skeletal
> muscle on T1, with intermediate-to-high heterogeneous signal on T2 fat-suppressed images.
> Look for dark non-enhancing collagen bands crossing the lesion and a linear tail extending
> along a fascial plane (fascial tail sign). Margins are infiltrative, not encapsulated.

*(49 words)*

### L — Lipomatous (lipoma, atypical lipomatous tumor / well-differentiated liposarcoma)

> Lipomatous tumor: a fat-containing mass that is uniformly bright on T1, matching
> subcutaneous fat, and turns dark on fat-suppressed T2 images. Well-circumscribed,
> homogeneous fat with only thin (under 2 mm) septa suggests lipoma; thick or nodular septa,
> non-fatty nodules, or persistent high T2 signal suggest atypical lipomatous
> tumor/well-differentiated liposarcoma.

*(49 words)*

### M — Myxoid (intramuscular myxoma, myxoid liposarcoma, myxofibrosarcoma)

> Myxoid tumor: a well-circumscribed, lobulated soft-tissue mass that is homogeneously
> darker than muscle on T1 and near-fluid-bright on T2 fat-suppressed images, sometimes
> mimicking a cyst. Look for a bright rim of perilesional fat or edema, internal fat droplets
> (myxoid liposarcoma), or, in myxofibrosarcoma, ill-defined infiltrative margins with a
> bright enhancing fascial tail.

*(52 words)*

### N — Peripheral nerve sheath (schwannoma, neurofibroma, MPNST)

> Nerve sheath tumor: a fusiform mass following a nerve, isointense to muscle on T1 and
> heterogeneously bright on T2. Look for a nerve entering and exiting the mass, a thin rim of
> fat at its poles (split-fat sign), a dark center with bright rim (target sign), or multiple
> ring-like fascicles. Absent signs with infiltrative margins suggest MPNST.

*(57 words)*

### V — Vascular tumors / malformations (hemangioma, venous malformation, angiosarcoma)

> Vascular lesion: a lobulated, multi-septated mass with serpiginous channels,
> isointense-to-hyperintense relative to muscle on T1 and strikingly bright on T2
> fat-suppressed images. Look for dark rounded phleboliths, fluid-fluid levels, and
> interspersed fat. Absent flow voids indicates slow flow (venous malformation, hemangioma);
> prominent flow voids indicates high flow.

*(47 words)*

---

## 2. Evidence section per group

Every claim in section 1 is traceable to one of the citations below. Quoted text is verbatim
from the cited source where shown in quotation marks; otherwise the finding is a close
paraphrase and is labeled as such.

### 2.1 F — Fibrous / fibroblastic–myofibroblastic

**Desmoid-type fibromatosis**

- T1 isointense to muscle + low-signal collagen bands; T2 signal falls as collagen matures;
  fascial tail in up to 90% of cases.
  Direct evidence (verbatim): "Fibromatoses heterogeneously enhance, are typically isointense
  in comparison with the signal intensity of skeletal muscle on T1-weighted MR images, and
  have bands of low signal intensity representative of crowded collagen bundles. Collagenized
  bands and linear extension along the fascial planes (ie, the 'fascial tail' sign) are seen
  in up to 90% of cases and are suggestive of this diagnosis."
  And: "On T2-weighted MR images, early-stage lesions with higher cellular components show
  high signal intensity. As the lesions mature and the collagen deposition increases, the
  signal intensity on T2-weighted MR images decreases."
  Source: Haseli S, Mansoori B, Christensen D, Abadi A, Pooyan A, Shomal Zadeh F, Mau B,
  Khalili N, Murphey M, Chalian M. *Fibroblastic and Myofibroblastic Soft-Tissue Tumors:
  Imaging Spectrum and Radiologic-Pathologic Correlation.* RadioGraphics 2023;43(9):e230005.
  doi:10.1148/rg.230005. https://pubs.rsna.org/doi/10.1148/rg.230005 (fetched and read;
  PMID 37440448).

- General rule linking T2 brightness to myxoid/necrotic content and T2 darkness to
  hypercellularity and dense collagen — supports the "heterogeneous T2" wording for
  fibrosarcoma/undifferentiated pleomorphic sarcoma (UPS).
  Direct evidence (verbatim, same source, abstract + teaching point): "lesions with a greater
  myxoid matrix or necrotic components tend to show higher signal intensity on T2-weighted MR
  images, whereas lesions with hypercellularity and dense internal collagen content display
  low signal intensity."

- **Solitary fibrous tumor (SFT)** — T2 is classically *variable* with a mixed black-and-white
  pattern from hypointense fibrous regions; low signal on both T1 and T2 reflects collagen or
  low cellularity; marked (hypervascular) enhancement.
  Direct evidence (verbatim): "On T2-weighted MR images, solitary fibrous tumors exhibit
  variable signal intensity, with a characteristic black and white mixed pattern
  corresponding to hypointense fibrous regions. Areas of low signal intensity on both T1- and
  T2-weighted MR images are attributable to collagen content or low cellularity."
  Source: same RadioGraphics 2023 article (rg.230005), Extrapleural Solitary Fibrous Tumor
  section.

- SFT is well-circumscribed rather than infiltrative (researcher inference from the same
  article's Figure 10 caption describing "a well-circumscribed mass" and marked heterogeneous
  enhancement). This is a caveat for the F description: the "infiltrative margins" clause
  applies to desmoid and to many sarcomas but not to SFT, which is typically circumscribed.

**Note on the "band sign" wording.** A search-summary source attributed a "band sign" label
(nonenhancing linear low signal on both T1 and T2) to Murphey MD, Ruble CM, Tyszko SM,
Zbojniewicz AM, Potter BK, Miettinen M. *Musculoskeletal Fibromatoses: Radiologic-Pathologic
Correlation.* RadioGraphics 2009;29(7):2143–2183. doi:10.1148/rg.297095138
(https://pubs.rsna.org/doi/10.1148/rg.297095138; PMID 19926768). **I could not retrieve the
full text of this article to verify the exact phrase, so the term "band sign" is not used in
the short description above.** The band-like low-signal collagen septa are independently
documented in rg.230005 (Figure 8 caption: "Axial STIR MR image shows a hyperintense lesion
with bandlike collagenous septa of low signal intensity").

### 2.2 L — Lipomatous

**Lipoma**

- Follows subcutaneous fat on all sequences: T1 high signal that suppresses on fat-saturated
  sequences; T2 high signal on FSE T2 that also suppresses ("persistent areas of high T2
  signal are worrisome"); no or minimal enhancement; thin capsule and very thin (< 2 mm)
  septations are common, whereas "avidly enhancing, thick/nodular septa or evidence of local
  invasion are suggestive of malignancy."
  Source: *Lipoma.* Radiopaedia.org (Radiology Reference Article; peer-reviewed reference
  article, community-maintained). https://radiopaedia.org/articles/lipoma (fetched and read).
  Support: direct evidence (verbatim). Note this is a reference-wiki source, not a journal
  review; used only for the fat-signal-following claim, which is also corroborated below.

**ALT / well-differentiated liposarcoma (WDLS)**

- Largely lipomatous mass (> 75% of the lesion) with non-lipomatous components.
  Direct evidence (verbatim, abstract): "Well-differentiated liposarcomas frequently
  demonstrate a diagnostic appearance on computed tomographic (CT) or magnetic resonance (MR)
  images, with a largely lipomatous mass (>75% of the lesion) and nonli[omatous components]…"
  Source: Murphey MD, Arcara LK, Fanburg-Smith J. *Imaging of Musculoskeletal Liposarcoma
  with Radiologic-Pathologic Correlation.* RadioGraphics 2005;25(5):1371–1395.
  doi:10.1148/rg.255055106. https://pubs.rsna.org/doi/10.1148/rg.255055106 (abstract verified
  via publisher page, doi.org and PubMed 16160117; full text partly paywalled).

- Thickened/nodular septa, non-adipose masses, prominent foci of high T2 signal, and
  prominent enhancement favour WDLS over simple lipoma.
  Direct evidence (verbatim): "Several imaging features are more closely associated with
  well-differentiated liposarcomas, including thickened or nodular septa, associated
  nonadipose masses, prominent foci of high T2 signal, and prominent areas of enhancement."
  This wording comes from an AJR article at https://ajronline.org/doi/10.2214/ajr.182.3.1820733
  = AJR Am J Roentgenol 2004;182(3):733–739, whose author list I could not resolve because the
  publisher page blocked fetching. The substantively equivalent, fully citable version of this
  claim is Ohguri T, Aoki T, Hisaoka M, Watanabe H, Nakamura K, Hashimoto H, Nakamura T,
  Nakata H. *Differential Diagnosis of Benign Peripheral Lipoma from Well-Differentiated
  Liposarcoma on MR Imaging: Is Comparison of Margins and Internal Characteristics Useful?*
  AJR Am J Roentgenol 2003;180(6):1689–1694. doi:10.2214/ajr.180.6.1801689 (PMID 12760945) —
  head-to-head MR comparison of 35 benign lipomas vs 23 WDLS, in which margins and internal
  characteristics (septa/nodularity) were the assessed discriminators.
  https://www.ajronline.org/doi/abs/10.2214/ajr.180.6.1801689

- Septa ≥ 2 mm and signal inhomogeneity were significantly associated with malignancy;
  sensitivity of "septa" was 37% with 100% specificity/PPV.
  Direct evidence (verbatim from results text): "the presence of septa >2 mm and
  inhomogeneity were significantly correlated with malignancy (p < 0.05)… The sensitivity of
  the sign 'septa' was 37%, with specificity of 100%, PPV of 100% and NPV of 21.6%."
  Source: *Magnetic Resonance Imaging Assessment of Lipomatous Soft-tissue Tumors.* In Vivo
  2017;31(3):389–397. PMC5461449. https://pmc.ncbi.nlm.nih.gov/articles/PMC5461449/ (fetched
  and read; 54 patients).

- General review framing (that lipomas are the most common benign fat-containing mass with a
  characteristic MR appearance) is from Gupta P, Potti TA, Wuertzer SD, Lenchik L, Pacholke
  DA. *Spectrum of Fat-containing Soft-Tissue Masses at MR Imaging: The Common, the Uncommon,
  the Characteristic, and the Sometimes Confusing.* RadioGraphics 2016;36(4):1257–1274.
  doi:10.1148/rg.2016150133 (abstract verified; full text paywalled; PMID 27163592).

### 2.3 M — Myxoid

**Intramuscular myxoma (the "bright as fluid" archetype)**

- Direct evidence (verbatim): "On MRI, the lesions are well circumscribed with smooth or
  slightly lobulated margins, homogeneously hypointense on T1-weighted images, and extremely
  hyperintense on T2-weighted images, similar to water."
  And: "The presence of a perilesional rind of fat or perilesional edema has been described
  to be highly suggestive of the diagnosis on MRI… A more prominent 'fat cap' may be present
  at the superior and inferior margins of the lesion."
  Also: "Myxomas are generally well defined and intramuscular, with whorled or lacy internal
  enhancement, and may show perilesional nonenhancing edema or a fat rind."
  Source: Baheti AD, Tirumani SH, Rosenthal MH, Howard SA, Shinagare AB, Ramaiya NH,
  Jagannathan JP. *Myxoid Soft-Tissue Neoplasms: Comprehensive Update of the Taxonomy and
  MRI Features.* AJR Am J Roentgenol 2015;204(2):374–385. doi:10.2214/AJR.14.12888.
  https://www.ajronline.org/doi/full/10.2214/AJR.14.12888 (fetched and read).

**Myxoid liposarcoma (MLPS)**

- Direct evidence (verbatim): "It appears as a well-defined multilobulated (often septate)
  heterogeneous mass… Its myxoid component appears low density on CT, T2-hyperintense and
  T1-hypointense on MRI, and enhances on contrast-enhanced studies… The fatty component
  appears T1 hyperintense, usually forms less than 10% of the total tumor volume, and may not
  always be visible on imaging."
  Source: Baheti et al., AJR 2015 (as above).

- The classic dedicated review notes MLPS may be mistaken for a cystic tumor when no contrast
  is given (i.e. it is nearly fluid-bright without fat signal).
  Direct evidence (verbatim, abstract): "Although myxoid liposarcoma is a subtype of
  liposarcoma, it may be difficult to establish the correct diagnosis with magnetic resonance
  (MR) imaging due to the lack of fat signal intensity. Without the administration of
  gadolinium contrast material, the tumor may even mimic a cystic tumor."
  Source: Sung MS, Kang HS, Suh JS, Lee JH, Park JM, Kim JY, Lee HG. *Myxoid Liposarcoma:
  Appearance at MR Imaging with Histologic Correlation.* RadioGraphics 2000;20(4):1007–1019.
  doi:10.1148/radiographics.20.4.g00jl021007 (abstract verified; PMID 10903690).
  https://pubs.rsna.org/doi/abs/10.1148/radiographics.20.4.g00jl021007

**Myxofibrosarcoma**

- Direct evidence (verbatim): "Lesions appear heterogeneous on both T1- and T2-weighted
  images. Lesions are T2 hyperintense with ill-defined infiltrative margins and inhomogeneous
  contrast enhancement. Although frequently T1 hypointense, areas of T1 isointensity to mild
  hyperintensity may be present. Because of their infiltrative nature, a T2-hyperintense
  enhancing curvilinear projection (tail) may be seen extending from the primary mass into
  the adjacent soft tissue ('tail sign')… Fluid-fluid levels may rarely be present."
  Source: Baheti et al., AJR 2015 (as above).

- Supportive (radiology–pathology): myxomas vs myxofibrosarcomas "have overlapping imaging
  features — namely, they are both hyperintense o[n T2]" and differ in normalised T1 signal.
  Source: *Radiomic features from MRI distinguish myxomas from myxofibrosarcomas.* PMC6694512.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6694512/ (search-result excerpt; not full-text
  verified).

### 2.4 N — Peripheral nerve sheath tumors

- Core signal pattern and the four classic signs.
  Direct evidence (verbatim, abstract): "On MRI, these tumors are seen as fusiform lesions,
  with low to intermediate signal intensity on T1W images and high-signal intensity on T2W
  images." The signs listed are "distribution of the tumor along a major nerve, an entering or
  exiting nerve sign, target sign, a fascicular sign and a split-fat sign."
  Source: Kakkar C, Shetty CM, Koteshwara P, Bajpai S. *Telltale signs of peripheral
  neurogenic tumors on magnetic resonance imaging.* Indian J Radiol Imaging
  2015;25(4):453–458. doi:10.4103/0971-3026.169447. PMC4693395 (fetched and read).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4693395/

- **Target sign** — direct evidence (verbatim): "The target sign is portrayed on axial T2W
  images as a central area of hypointensity with a peripheral hyperintensity. This sign is
  most often seen with neurofibromas… Pathologically, there is a central fibrocollagenous
  core and a surrounding myxomatous tissue." For schwannoma the basis is "a central
  distribution of the more cellular Antoni type A cells, with a surrounding rim of
  hypocellular Antoni type B cells."
  Source: Kakkar et al., above.

- **Fascicular sign** — direct evidence (verbatim): "represented by multiple ring-like
  structures, which appear as hypointense foci within the hyperintense area on T2W images,
  possibly reflecting the fascicular bundles seen histologically… This sign is often seen in
  benign nerve sheath tumors."

- **Split-fat sign** — direct evidence (verbatim): "a 'split fat sign' refers to the presence
  of fat at the upper and lower poles of a lesion on T1W images, which is suggestive of the
  intermuscular location of the lesion… It is a feature of benignity, as malignant lesions
  tend to be more infiltrative in nature, resulting in the obliteration of fat at the ends of
  the lesion."

- **Entering/exiting nerve ("tail sign")** — direct evidence (verbatim): "the involved nerve
  can be seen entering and/or exiting the neoplasm, resembling a tail coming off the tumor.
  Virtually pathognomonic for PNSTs… This feature is seen in both benign PNSTs (BPNSTs) and
  MPNSTs."
  Source: *Advanced MR Imaging of Peripheral Nerve Sheath Tumors Including Diffusion Imaging.*
  Semin Musculoskelet Radiol 2015. doi:10.1055/s-0035-1546823 (search-result excerpt, verbatim).

- **MPNST red flags** — perilesional edema and bony destruction favour malignancy; target
  sign within a malignant lesion suggests residual benign tissue.
  Direct evidence (verbatim): "the presence of perilesional edema that is seen as an area of
  increased signal intensity on T2W images, adjacent to the primary lesion, is suggestive of
  perilesional infiltration or edema… This feature is usually absent in benign neurogenic
  lesions." And: "The presence of this sign [target sign] in a malignant peripheral nerve
  sheath tumor is indicative of benign tissue within the lesion, however, the absence of this
  sign in a lesion is indicative of a malignant transformation of the primary benign lesion."
  Source: Kakkar et al., above.

- **Prevalence in an intramuscular PNST cohort** (supportive, primary data): 30
  pathologically confirmed benign intramuscular PNSTs (24 schwannomas, 3 ancient schwannomas,
  3 neurofibromas). "All the patients who had intramuscular schwannomas" showed split-fat
  sign, low-signal margin, and fascicular sign; all 3 ancient schwannomas showed only the
  split-fat sign among classic signs; all 3 neurofibromas showed split-fat sign and one had a
  target sign.
  Source: Lee SK, Kim JY, Lee YS, Jeong HS. *Intramuscular peripheral nerve sheath tumors:
  schwannoma, ancient schwannoma, and neurofibroma.* Skeletal Radiol 2020;49(6):967–975.
  doi:10.1007/s00256-020-03371-6 (PMID 31932870). Results text read via
  https://link.springer.com/article/10.1007/s00256-020-03371-6

- Authoritative imaging–pathology reference for the signs (used in the source-check below):
  Murphey MD, Smith WS, Smith SE, Kransdorf MJ, Temple HT. *Imaging of Musculoskeletal
  Neurogenic Tumors: Radiologic-Pathologic Correlation.* RadioGraphics
  1999;19(5):1253–1280. doi:10.1148/radiographics.19.5.g99se101253 (PMID 10489179). Also
  Beaman FD, Kransdorf MJ, Menke DM. *Schwannoma: Radiologic-Pathologic Correlation.*
  RadioGraphics 2004;24(5):1477–1481. doi:10.1148/rg.245045001 (PMID 15371621).

### 2.5 V — Vascular tumors and malformations

**Venous malformation (slow-flow)**

- Direct evidence (verbatim): "Pure venous malformations appear as lobulated and multi-septated
  masses with internal serpentine architecture. They exhibit intermediate to low signal
  intensity on T1- and high signal on T2-WI. The presence of internal hemorrhage or
  proteinaceous content may lend higher signal on T1WI or fluid-fluid levels. When present,
  phleboliths are diagnostic of venous elements."
  Also (from the MR-features table): venous malformation = "Intermediate T1, bright T2, absent
  flow voids, phleboliths/areas of thrombosis, localized or infiltrative, enhances gradually
  with contrast, normal feeding arteries and draining veins."
  Source: Bashir U, Shah S, Jeph S, O'Keeffe M, Khosa F. *Magnetic Resonance (MR) Imaging of
  Vascular Malformations.* Pol J Radiol 2017;82:731–741. doi:10.12659/PJR.903491. PMC5894044
  (fetched and read). https://pmc.ncbi.nlm.nih.gov/articles/PMC5894044/

- Fluid-fluid levels in venous malformations: historically considered uncharacteristic, but
  in a 40-patient series they were seen in 55% of newer MRIs vs 5% of older MRIs (p = 0.001);
  phleboliths were seen in 46.7% overall. Conclusion (verbatim): "FFLs should no longer be
  considered exclusionary for the diagnosis of VMs."
  Source: Alexander MD, et al. *Revisiting classic MRI findings of venous malformations:
  Changes in protocols may lead to potential misdiagnosis.* PMC6136139 (fetched and read).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6136139/

- Slow-flow channels appear as "T2-hyperintense lobules or serpiginous channels" with delayed
  heterogeneous enhancement; phleboliths are often the distinguishing feature between an AVM
  and a hemangioma, "the latter of which tends to contain intralesional fat."
  Source: Shyam K, Cicilet S, Philip B. *Among the fibers: A multimodality imaging review of
  intramuscular mass lesions.* Indian J Radiol Imaging 2018;28(2):214–224.
  doi:10.4103/ijri.IJRI_299_17. PMC6038209 (fetched and read).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6038209/

**Intramuscular hemangioma**

- Direct evidence (verbatim): "The vascular channels are seen as a tangle of tubular
  structures with hypointense to isointense signal on T1-weighted images and hyperintense
  signal on T2-weighted images relative to muscle. The flow voids seen on all pulse sequences
  indicate rapid blood flow. On T1-weighted images, fat overgrowth is seen as areas of
  hyperintensity. Phleboliths appear as circular areas of hypointensity. Haemorrhage may be
  detected as areas of hyperintense signal on T1-weighted images, with fluid-fluid levels on
  T2-weighted images."
  Also: "Phleboliths, which are characteristic calcifications present in cavernous and venous
  haemangiomas, are seen in 20% of intramuscular haemangiomas."
  Source: Pattamapaspong N, Peh WCG, Shek TWH. *Imaging of intramuscular haemangiomas of the
  extremities.* Singapore Med J 2020;61(3):122–128. doi:10.11622/smedj.2020030. PMC7905120
  (fetched and read). https://pmc.ncbi.nlm.nih.gov/articles/PMC7905120/

- "Presence of fat within a hemangioma is characteristic, and often serves to differentiate it
  from an intramuscular arteriovenous malformation (the latter of which tends to exhibit
  phleboliths)."
  Source: Shyam et al., Indian J Radiol Imaging 2018 (as above).

- High-flow malformations (AVM/AVF) show "dilated tortuous vessels and flow voids on spin echo
  images"; MRI classification into low-flow vs high-flow is a key, treatment-relevant
  determination.
  Source: Bashir et al., Pol J Radiol 2017 (as above).

- Background review for soft-tissue vascular malformations (classification, therapy follow-up):
  Flors L, Leiva-Salinas C, Maged IM, et al. *MR Imaging of Soft-Tissue Vascular Malformations:
  Diagnosis, Classification, and Therapy Follow-up.* RadioGraphics 2011;31(5):1321–1340.
  doi:10.1148/rg.315105213 (PMID 21918047). Cited here as the canonical RSNA review; not
  full-text retrieved.

---

## 3. Statements that are uncertain or vary by subtype

1. **Category membership mismatch (dataset vs task wording).** MSTT-199's "Fibrous" category
   is desmoid fibromatosis + undifferentiated pleomorphic sarcoma, and its "Fat" category is
   WDLS + dedifferentiated liposarcoma; it does not list lipoma, SFT, fibrosarcoma,
   neurofibroma, myxoid liposarcoma, or myxofibrosarcoma by those names. The short
   descriptions were written for the task's broader entity lists. If the prompt must match
   MSTT-199 exactly, the F description's "infiltrative margins" clause is appropriate for
   desmoid and UPS but wrong for SFT, and the L description's "fat-dominant" framing may not
   match dedifferentiated liposarcoma, whose dedifferentiated component is non-fatty.
   (Direct evidence for dataset composition: arXiv:2409.03110.)

2. **Dedifferentiated liposarcoma (in MSTT-199 "Fat").** The dedifferentiated component is a
   non-lipomatous solid mass (intermediate T1, heterogeneous T2-hyperintense), so a
   fat-only description can miss it. The > 75% lipomatous figure applies specifically to
   well-differentiated liposarcoma (Murphey et al., RadioGraphics 2005). Researcher inference:
   for L, a model scanning only for fat signal may fail on dedifferentiated regions.

3. **Angiosarcoma (in MSTT-199 "Vascular").** Angiosarcoma is a malignant vascular neoplasm
   that may be heterogeneous with necrosis and hemorrhage and need not show phleboliths,
   organised serpentine slow-flow channels, or the classic interspersed fat of a hemangioma.
   No source retrieved in this pass specifically describes angiosarcoma MRI in these terms;
   therefore the V description's slow-flow/hemangioma features may under-cover angiosarcoma.
   **This is an acknowledged evidence gap.**

4. **Target sign is mostly a neurofibroma sign.** Kakkar et al. state the target sign "is most
   often seen with neurofibromas," but MSTT-199's "Nerve" category is schwannoma + MPNST.
   In the intramuscular PNST cohort, target sign was seen in only 1 of 3 neurofibromas and
   was not part of the classic profile for the 24 schwannomas (Lee et al., Skeletal Radiol
   2020). Researcher inference: within MSTT-199, the split-fat sign, fascicular sign,
   thin T2-hyperintense rim, and entering/exiting nerve are more reliable than the target sign.

5. **Split-fat sign is not specific to nerve sheath tumors** and is a feature of benignity;
   it can occur in other intermuscular soft-tissue tumors, and malignant lesions obliterate
   polar fat (Kakkar et al. 2015). So it should not be treated as pathognomonic.

6. **Myxoid groups overlap heavily.** Myxoma vs myxofibrosarcoma vs myxoid liposarcoma share
   near-fluid T2 brightness; myxoma is generally well-defined/intramuscular with a
   perilesional fat rind, myxofibrosarcoma is infiltrative with a fascial "tail," and myxoid
   liposarcoma carries small amounts of fat (< 10% of volume, possibly invisible). Imaging
   features overlap and biopsy remains the reference standard (Baheti et al., AJR 2015;
   Sung et al., RadioGraphics 2000). Researcher inference: no single sign reliably separates
   these three on one slice.

7. **T1/T2 signal of desmoid varies with lesion maturity.** Early, cellular desmoids are T2
   hyperintense; mature, collagenized desmoids become T2 hypo/intermediate (Haseli et al.,
   RadioGraphics 2023). The short description says "intermediate-to-high heterogeneous" to
   cover both, but a very collagenized desmoid can look T2-dark — this was a recognized
   failure mode in MSTT-199 itself, where "small fibrous tissue structure [was] not easily
   distinguishable from surrounding tissue."

8. **Phlebolith detection is modality- and sequence-dependent.** Phleboliths appear as
   low-signal foci on MRI but are better seen at CT; calcification/phleboliths "may also
   appear as flow-voids and can lead to erroneous labelling" (Bashir et al., Pol J Radiol
   2017). Also, only ~20% of intramuscular hemangiomas contain phleboliths (Pattamapaspong
   et al. 2020), so absence of phleboliths is not evidence against a vascular lesion.

9. **Signal descriptions are relative and field/protocol dependent.** All quoted statements
   compare signal to skeletal muscle and subcutaneous fat. Fat-suppressed T2 obscures the
   "interspersed fat" sign of hemangioma and the fatty component of myxoid liposarcoma,
   which is why MSTT-199 annotated fat tumors on T1 (arXiv:2409.03110). The V description
   relying on "interspersed fat" therefore works best on the T1 image.

10. **Evidence limitation for two RSNA full texts.** The full text of RadioGraphics
    rg.297095138 (Musculoskeletal Fibromatoses) and rg.255055106 (Musculoskeletal
    Liposarcoma) could not be retrieved (publisher paywall/blocked); only abstracts and search
    excerpts were available. Claims from those articles are labeled as abstract-level or
    excerpt-level evidence.

---

## 4. Source validation note

`source_check` was invoked for two decision-critical claims: (a) the desmoid collagen-band /
fascial-tail / T1-iso / T2-maturation claim, and (b) the PNST signal-pattern-and-signs claim.
In both cases the tool returned `status: unclear (confidence 0.30)` with the rationale
"automated semantic support or contradiction assessment is unavailable; review the cited
passages manually." It did, however, return relevant source lists. **Validation limitation:
automated claim-level verification was not available in this environment.** In lieu of that,
each key claim above was verified by fetching and reading the original source text directly
(the RadioGraphics 2023 fibroblastic-tumors article, the AJR 2015 myxoid-neoplasms article,
and the Kakkar et al. 2015 PMC full text were all fetched and quoted verbatim). Quoted
passages are marked "direct evidence (verbatim)"; paraphrases are explicitly labeled.

---

## 5. Citation list

1. Reasat T, Chenard S, Rekulapelli A, Chadwick N, Shechtel J, van Schaik K, Smith DS,
   Lawrenz J. *MSTT-199: MRI Dataset for Musculoskeletal Soft Tissue Tumor Segmentation.*
   arXiv:2409.03110 [eess.IV]. 4 Sep 2024. https://arxiv.org/abs/2409.03110
   — dataset composition per category, sequences acquired, annotation modality.
2. Haseli S, Mansoori B, Christensen D, Abadi A, Pooyan A, Shomal Zadeh F, Mau B, Khalili N,
   Murphey M, Chalian M. *Fibroblastic and Myofibroblastic Soft-Tissue Tumors: Imaging
   Spectrum and Radiologic-Pathologic Correlation.* RadioGraphics 2023;43(9):e230005.
   doi:10.1148/rg.230005. https://pubs.rsna.org/doi/10.1148/rg.230005
   — desmoid T1/T2 signal, collagen bands, fascial tail, SFT, fibrosarcoma, myxofibrosarcoma.
3. Murphey MD, Ruble CM, Tyszko SM, Zbojniewicz AM, Potter BK, Miettinen M. *Musculoskeletal
   Fibromatoses: Radiologic-Pathologic Correlation.* RadioGraphics 2009;29(7):2143–2183.
   doi:10.1148/rg.297095138. https://pubs.rsna.org/doi/10.1148/rg.297095138
   — fibromatosis reference (full text not retrieved).
4. Murphey MD, Arcara LK, Fanburg-Smith J. *Imaging of Musculoskeletal Liposarcoma with
   Radiologic-Pathologic Correlation.* RadioGraphics 2005;25(5):1371–1395.
   doi:10.1148/rg.255055106. https://pubs.rsna.org/doi/10.1148/rg.255055106
   — WDLS largely lipomatous (> 75%) with non-lipomatous components.
5. Gupta P, Potti TA, Wuertzer SD, Lenchik L, Pacholke DA. *Spectrum of Fat-containing
   Soft-Tissue Masses at MR Imaging: The Common, the Uncommon, the Characteristic, and the
   Sometimes Confusing.* RadioGraphics 2016;36(4):1257–1274. doi:10.1148/rg.2016150133.
   https://pubs.rsna.org/doi/10.1148/rg.2016150133 — fat-containing masses overview.
6. Ohguri T, Aoki T, Hisaoka M, Watanabe H, Nakamura K, Hashimoto H, Nakamura T, Nakata H.
   *Differential Diagnosis of Benign Peripheral Lipoma from Well-Differentiated Liposarcoma on
   MR Imaging: Is Comparison of Margins and Internal Characteristics Useful?* AJR Am J
   Roentgenol 2003;180(6):1689–1694. doi:10.2214/ajr.180.6.1801689 (PMID 12760945).
   https://www.ajronline.org/doi/abs/10.2214/ajr.180.6.1801689 — lipoma vs WDLS MR features.
7. *Magnetic Resonance Imaging Assessment of Lipomatous Soft-tissue Tumors.* In Vivo
   2017;31(3):389–397. PMC5461449. https://pmc.ncbi.nlm.nih.gov/articles/PMC5461449/
   — septa > 2 mm and inhomogeneity associated with malignancy.
8. *Lipoma.* Radiopaedia.org. https://radiopaedia.org/articles/lipoma
   — lipomas follow subcutaneous fat signal on all sequences; thin (< 2 mm) septa.
9. Baheti AD, Tirumani SH, Rosenthal MH, Howard SA, Shinagare AB, Ramaiya NH, Jagannathan JP.
   *Myxoid Soft-Tissue Neoplasms: Comprehensive Update of the Taxonomy and MRI Features.*
   AJR Am J Roentgenol 2015;204(2):374–385. doi:10.2214/AJR.14.12888.
   https://www.ajronline.org/doi/full/10.2214/AJR.14.12888
   — myxoma (T1 hypo, T2 water-bright, perilesional fat rind), MLPS (myxoid + <10% fat),
   myxofibrosarcoma (infiltrative, tail sign).
10. Sung MS, Kang HS, Suh JS, Lee JH, Park JM, Kim JY, Lee HG. *Myxoid Liposarcoma:
    Appearance at MR Imaging with Histologic Correlation.* RadioGraphics
    2000;20(4):1007–1019. doi:10.1148/radiographics.20.4.g00jl021007 (PMID 10903690).
    https://pubs.rsna.org/doi/abs/10.1148/radiographics.20.4.g00jl021007
    — MLPS may mimic a cystic tumor without contrast.
11. Kakkar C, Shetty CM, Koteshwara P, Bajpai S. *Telltale signs of peripheral neurogenic
    tumors on magnetic resonance imaging.* Indian J Radiol Imaging 2015;25(4):453–458.
    doi:10.4103/0971-3026.169447 (PMID 26752825). PMC4693395.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC4693395/ — target, fascicular, split-fat,
    entering/exiting nerve signs; MPNST features.
12. Murphey MD, Smith WS, Smith SE, Kransdorf MJ, Temple HT. *Imaging of Musculoskeletal
    Neurogenic Tumors: Radiologic-Pathologic Correlation.* RadioGraphics
    1999;19(5):1253–1280. doi:10.1148/radiographics.19.5.g99se101253 (PMID 10489179).
    — canonical PNST signs reference.
13. Beaman FD, Kransdorf MJ, Menke DM. *Schwannoma: Radiologic-Pathologic Correlation.*
    RadioGraphics 2004;24(5):1477–1481. doi:10.1148/rg.245045001 (PMID 15371621).
14. Lee SK, Kim JY, Lee YS, Jeong HS. *Intramuscular peripheral nerve sheath tumors:
    schwannoma, ancient schwannoma, and neurofibroma.* Skeletal Radiol 2020;49(6):967–975.
    doi:10.1007/s00256-020-03371-6 (PMID 31932870).
    https://link.springer.com/article/10.1007/s00256-020-03371-6 — prevalence of classic signs.
15. Bashir U, Shah S, Jeph S, O'Keeffe M, Khosa F. *Magnetic Resonance (MR) Imaging of
    Vascular Malformations.* Pol J Radiol 2017;82:731–741. doi:10.12659/PJR.903491.
    PMC5894044. https://pmc.ncbi.nlm.nih.gov/articles/PMC5894044/ — venous malformation
    lobulated/multi-septated/serpentine, T1 intermediate-low, T2 bright, phleboliths.
16. Flors L, Leiva-Salinas C, Maged IM, et al. *MR Imaging of Soft-Tissue Vascular
    Malformations: Diagnosis, Classification, and Therapy Follow-up.* RadioGraphics
    2011;31(5):1321–1340. doi:10.1148/rg.315105213 (PMID 21918047) — canonical RSNA
    vascular-malformation review.
17. Alexander MD, et al. *Revisiting classic MRI findings of venous malformations: Changes in
    protocols may lead to potential misdiagnosis.* PMC6136139.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC6136139/ — fluid-fluid levels in VMs.
18. Pattamapaspong N, Peh WCG, Shek TWH. *Imaging of intramuscular haemangiomas of the
    extremities.* Singapore Med J 2020;61(3):122–128. doi:10.11622/smedj.2020030. PMC7905120.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC7905120/ — hemangioma MR signal, phleboliths,
    interspersed fat, flow voids.
19. Shyam K, Cicilet S, Philip B. *Among the fibers: A multimodality imaging review of
    intramuscular mass lesions.* Indian J Radiol Imaging 2018;28(2):214–224.
    doi:10.4103/ijri.IJRI_299_17. PMC6038209.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC6038209/ — hemangioma fat vs AVM phleboliths;
    slow-flow VM T2-hyperintense serpiginous channels.
20. *Advanced MR Imaging of Peripheral Nerve Sheath Tumors Including Diffusion Imaging.*
    Semin Musculoskelet Radiol 2015. doi:10.1055/s-0035-1546823 — entering/exiting nerve
    ("tail") sign; seen in both benign PNSTs and MPNSTs.

---

## 6. Sources considered and rejected/deprioritized

- *AJR 182(3):733 (ajronline.org/doi/10.2214/ajr.182.3.1820733)* — quoted a useful WDLS-feature
  sentence, but the article's author list could not be resolved because the publisher page
  blocked fetching. Replaced by Ohguri et al. AJR 2003 for the same claim.
- *Radiopaedia "Lipoma"* — community-maintained reference wiki, not a journal review; used
  only for the fat-signal-following claim that is corroborated by journal sources.
- *arXiv HTML mirrors "bytez.com", "selectdataset.com", "emergentmind.com"* for MSTT-199 —
  deprioritized in favour of the primary arXiv paper.
- *"Soft-Tissue Tumors and Tumorlike Lesions: A Systematic Imaging Approach"* — search
  snippet attributed a useful lipoma figure caption to a DOI (10.1148/radiol.2532081199) whose
  journal/title mapping could not be confirmed; not cited.
- *Myxoid-liposarcoma Wikipedia page* — deprioritized (secondary, unverified clinical claims).
- *PMC13198472, PMC11246874, medRxiv/PubMed 2026 LLM-segmentation papers* — surfaced during
  the "practical experience" search angle; relevant to the broader project but not needed to
  support MRI-appearance claims, so not used.

---

## 7. Next steps (most useful follow-ups)

1. Confirm the exact MSTT-199 entity inventory per category from the released dataset metadata
   (the paper names only one benign and one malignant entity per category; the released
   labels may be finer-grained). This determines whether the L description needs a
   dedifferentiated-liposarcoma clause and whether F needs an SFT clause.
2. Retrieve full text of RadioGraphics rg.297095138 and rg.255055106 to upgrade the "band
   sign" and WDLS (> 75% lipomatous) claims from excerpt/abstract-level to full-text-verified.
3. For angiosarcoma (MSTT-199 Vascular), locate a dedicated imaging review to add a malignant
   vascular description, since no source in this pass supported it.
4. Optionally add a T1-hyperintense-hemorrhage caveat for F/V groups (subacute hemorrhage is
   T1-bright and is not fat), which matters if the model must not confuse blood with fat on T1.
