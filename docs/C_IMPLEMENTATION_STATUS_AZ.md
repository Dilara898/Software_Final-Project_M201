# C hissəsi: implementasiya statusu

## Tamamlanan müstəqil işlər

- Mənbələrin paralel toplanması, paylaşılmış semaphore və client ownership.
- Keş oxuma/yazma, növbə və fetch üçün ayrı timeout büdcələri.
- Keş yazma timeout-u uğurlu mənbəni silmir; regression testi mövcuddur.
- Cancellation cleanup, sabit mənbə sırası və qismən uğursuzluq nəticələri.
- Typed portlar və nəticələr; A-nın modelləri və B-nin servisi yaradılmayıb.
- Research axını: collect → synthesize → citation validation → history.
- Tarixçə nasazlığında cavab qorunur; timeout nəticəsi unknown kimi göstərilir.
- Reproduksiya metadatası, ayrıca setup/teardown ölçümü və xam CSV ilə benchmark.
- Bir/iki/üç mənbə nasazlığının offline demonstrasiyası.

## Tarixi yoxlama nəticələri (2026-09-16, dərin review-dan əvvəl)

Bu bölmə ilkin yoxlamanın tarixçəsidir. Dərin review mərhələsinin nəticəsi 2026-09-17 yoxlamasına
əsasən 116 C testi və ayrıca 16 orijinal smoke-test olmaqla cəmi 132 testdir.

- 101 C testi + dəyişdirilməmiş 16 AI smoke-testi: **117 passed**.
- `researcher.concurrency` statement/branch coverage: **99%** (yuvarlaqlaşdırılmış).
- Ruff: keçdi. mypy: 9 source faylında xəta yoxdur.
- Python 3.14 Windows runtime-da yoxlanılıb; 9 source faylı Python 3.11
  qrammatikası ilə parse edilib. Bu, Python 3.11 runtime testi demək deyil.
- `ai/` daxilində 10 Python faylı orijinal paketlə mətn olaraq eynidir.
- Test temporary qovluğu workspace `.cache/` altında verilir; sistem temporary
  qovluğuna sandbox giriş məhdudiyyəti test kodu dəyişdirilmədən həll edilib.
- Orijinal smoke-testlər yoxlama üçün `.cache/ai-contract/tests/` altına
  dəyişdirilmədən köçürülüb; shared test faylları əvəz edilməyib.

## Offline benchmark

5 sual × 3 mənbə × 3 təkrar, hər iki rejimdə keş söndürülüb:

| Ölçü | Nəticə |
|---|---:|
| Sequential median | 942.253 ms |
| Parallel median | 467.588 ms |
| Nisbət | 2.015x |

Bu nəticə süni 40/80/40 ms gecikmələrlə ölçülüb. Real API, LLM və DB
performansı haqqında nəticə çıxarmaq olmaz. Xam ölçülər
`artefacts/benchmark-offline-v2.csv`, hesabat `.md` faylındadır.
Tarixi ilk ölçmə saxlanılıb.

## Komanda inteqrasiyasından asılı qalan işlər

1. A-nın Settings/cache/history adapterlərinin bağlanması və real DB testləri.
2. B-nin async AIService, retry, provider pacing və SDK timeout yoxlamaları.
3. D-nin CLI/Docker inteqrasiyası və istifadəçiyə warning/error göstərilməsi.
4. Real providerlərlə benchmark, SDK/DB daxil olmaqla end-to-end yoxlama.
5. Komanda hesabatı və müdafiə materiallarında real nəticələrin əlavə edilməsi.

Ətraflı portlar, ownership və reproduksiya əmrləri [ROLE_C.md](ROLE_C.md)-dədir.
Bu mərhələ C-nin müstəqil implementasiyasıdır; real A/B/D inteqrasiyasının
və bütün tətbiqin hazır olduğu iddia edilmir.

## Təhvil və reproduksiya

Kod və offline artefaktlar `feat/c-concurrency-benchmark` branch-indədir.
Bu branch komanda review-u üçün təqdim edilir; real inteqrasiya ayrıca mərhələdir.
Dərin review mərhələsində repoda 116 C testi var idi. Cari say aşağıdakı 2026-09-18 yoxlama bölməsindədir.
Həmin mərhələnin 132 test nəticəsi 116 C testindən və ayrıca lokalda yoxlanmış
16 orijinal smoke-testdən ibarətdir. Orijinal smoke-testlər hazırkı branch-in
test fayllarına daxil deyil. Əvvəlki 101 C / 117 ümumi nəticəsi yalnız
2026-09-16 tarixli ilkin yoxlamaya aiddir.

## Dərin review-dan sonrakı yoxlama (2026-09-17)

- 116 C testi və ayrıca 16 dəyişdirilməmiş orijinal smoke-test keçdi: cəmi 132.
- C statement/branch coverage yenə 99%-dir; Ruff və mypy keçdi.
- Yeni 15 regression ssenarisi cancellation cleanup, warning logları, URL portları,
  cache nəticə limiti, böyük rəqəmli istinad, live client timeout-u və benchmark
  giriş faylının qorunmasını yoxlayır.
- İlkin altı review problemi və əlavə giriş faylının üzərinə yazılma problemi
  düzəldilib. Real A/B/D inteqrasiyası və live ölçmə yenə ayrıca yoxlanmalıdır.

## A/D ilə uyğunlaşdırma (tarixi nəticə: main e49b8eb əsasında)

- A-nın cache və history funksiyalarını bağlayan C adapterləri əlavə edilib;
  shared A/B/D faylları dəyişdirilməyib.
- D-nin ayrıca feat/d-cli branch-indəki parser ilə wiki → wikipedia və
  --no-cache mapping-i offline yoxlanıb. Həmin branch main-ə merge edilməyib.
- Həmin mərhələdə 16 uyğunluq testi ilə C test sayı 132 idi. A-nın 4 cache testi ilə
  birlikdə seçilmiş suite-də 136 test keçib. Bunlar əvvəlki 116 C + 16 orijinal
  smoke-test nəticəsi ilə eyni say hesabı deyil.
- A-nın config testində import-time Settings xətası qalır; bütün ümumi suite-in
  problemsiz keçdiyi iddia edilmir. B-nin AIService və exceptions modulları,
  tətbiqin real CLI lifecycle-ı və real DB/API ölçmələri hələ gözlənilir.

## Main və A/B/D ilə uyğunlaşdırma (2026-09-18, ilk yoxlama)

- `feat/c-storage-integration` son `origin/main` (`386f3d4`) üzərinə
  fast-forward edilib. B-nin servis/retry/exception/business-logic düzəlişləri
  və D parser-i artıq lokaldadır; D-nin ayrıca entrypoint/Docker branch-ləri
  bu main snapshot-ına daxil deyil.
- `sources_from_cli` B-nin ortaq `select_sources` funksiyasına bağlanıb:
  alias, dedup, canonical sıra və ValidationError eyni siyasətdən gəlir.
- A storage adapterləri, B-nin real servisləri və verilmiş AI funksiyaları
  ilə 20 inteqrasiya testi keçir. HTTP MockTransport, fake LLM və fake DB
  istifadə olunur; real şəbəkə/SDK/PostgreSQL işləyişi iddia edilmir.
- Cari C test sayı **136**-dır: əvvəlki 116 + 17 storage/source adapter testi
  + 3 B/C workflow testi. Yeni workflow sınaqları cache hit/bypass, Wikipedia
  summary retry, arXiv 404 degradation, citations/history və bütün mənbələrin
  uğursuzluğunda LLM çağırılmamasını yoxlayır.
- Tam repository suite: **228 passed, 1 failed**. Qalan uğursuz test
  `tests/test_config.py::test_missing_database_url_raises`-dır: A config
  import zamanı Settings yaratdığı üçün testin raises blokuna çatmır.
  Test gizlədilməyib və A-nın faylı bu uyğunlaşdırmada dəyişdirilməyib.
- C kodu/scriptləri və yeni inteqrasiya testləri üçün Ruff keçib;
  C-nin 7 moduluna mypy yoxlaması keçib. Yeni coverage faizi ölçülməyib;
  yuxarıdakı 99% əvvəlki mərhələnin nəticəsidir.
- B synthesis servisi tətbiq teardown-unda `shutdown()` tələb edir.
  ROLE_C composition nümunəsi iki ayrı B servisi və bu ownership ilə yenilənib.
- Qalan iş: D-nin tam application lifecycle-ı, A config problemi və real
  DB/API/Docker yoxlamaları. B factory-sində pacing default olaraq söndürülüb;
  production/live benchmark üçün interval ayrıca verilməlidir.

## Son yoxlama: A düzəlişindən sonra (2026-09-18)

- Son `main` (`7f10a7b`) lokal inteqrasiya branch-inə fast-forward edilib.
  A Settings-i lazy `get_settings()` ilə yükləyir; əvvəlki config import
  uğursuzluğu həll edilib. C adapterləri Settings və pool yaratmadığından
  onların kodunda əlavə dəyişiklik tələb olunmayıb.
- **Tam offline suite: 231 passed**, uğursuz test yoxdur. Bu sayın **136-sı
  C testidir**, o cümlədən 20 A/B/C/D müqavilə uyğunluğu testi.
- C modulları/scriptləri və yeni testlər üçün Ruff, C-nin 7 modulu üçün
  mypy keçib. `git diff --check` keçib. Real DB/API/Docker yoxlaması deyil.
- Yuxarıdakı 228 passed / 1 failed yalnız `386f3d4` üzərində əvvəlki
  yoxlamanın tarixi nəticəsidir. Cari config blokeri qalmayıb.
- Tətbiq startup-ı `get_settings()` çağırmalı, lifecycle sonunda B synthesis
  servisini, HTTP client-i və A pool-unu sahiblik qaydasına uyğun bağlamalıdır.
  D-nin main-ə daxil olmayan işləri və canlı sistem yoxlamaları ayrıca qalır.
