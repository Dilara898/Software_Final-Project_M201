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

## Yoxlama nəticələri (2026-09-16)

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
Repo daxilindəki 101 C testini ROLE_C.md-dəki əmrlə işlətmək mümkündür.
Yuxarıdakı 117 nəticəsinə əlavə 16 orijinal smoke-testin lokal yoxlaması daxildir;
onlar hazırkı branch-in test fayllarına daxil deyil.
