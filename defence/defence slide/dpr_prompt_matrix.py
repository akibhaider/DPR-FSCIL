"""Pairwise discriminative prompt matrices M (n x n x k) for DPR.

M[i][j] holds k=3 prompts that describe class i while contrasting it against
class j. The matrix is directional: M[i][j] != M[j][i]. Diagonal entries are
None (a class is never discriminated against itself).

Each of the k prompts for a pair targets a *different* distinguishing attribute,
so averaging their text-encoder embeddings yields a class-pair discriminative
prototype N[i][j] that spans several discriminative cues rather than repeating one.

Class choices are deliberately the hardest confusable groups in each dataset,
since DPR only activates on pairs whose similarity exceeds r_th:
  CIFAR-100     : the full "large carnivores" superclass
  miniImageNet  : five hound/gundog breeds from the train split
  CUB-200-2011  : five gull species (classes 059-064)
"""

K = 3

CLASSES = {
    "cifar100": ["bear", "leopard", "lion", "tiger", "wolf"],
    "miniimagenet": [
        "Walker hound",
        "Ibizan hound",
        "Saluki",
        "Gordon setter",
        "golden retriever",
    ],
    "cub200": [
        "California Gull",
        "Glaucous-winged Gull",
        "Heermann's Gull",
        "Herring Gull",
        "Ring-billed Gull",
    ],
}

# WordNet ids / CUB folder names for the classes above, for pipeline lookup.
CLASS_IDS = {
    "miniimagenet": {
        "Walker hound": "n02089867",
        "Ibizan hound": "n02091244",
        "Saluki": "n02091831",
        "Gordon setter": "n02101006",
        "golden retriever": "n02099601",
    },
    "cub200": {
        "California Gull": "059.California_Gull",
        "Glaucous-winged Gull": "060.Glaucous_winged_Gull",
        "Heermann's Gull": "061.Heermann_Gull",
        "Herring Gull": "062.Herring_Gull",
        "Ring-billed Gull": "064.Ring_billed_Gull",
    },
}

PROMPTS = {
    "cifar100": {
        ("bear", "leopard"): [
            "a photo of a bear, whose thick shaggy fur is a single plain color with none of the rosette spots of a leopard",
            "a photo of a bear, with a heavy barrel shaped body far bulkier than the sleek frame of a leopard",
            "a photo of a bear, showing a short stubby tail instead of the long trailing tail of a leopard",
        ],
        ("bear", "lion"): [
            "a photo of a bear, with a broad bare faced head that never carries the thick mane of a lion",
            "a photo of a bear, whose small rounded ears sit high on the skull unlike the flat ears of a lion",
            "a photo of a bear, standing flat on its whole foot rather than up on its toes like a lion",
        ],
        ("bear", "tiger"): [
            "a photo of a bear, whose uniform dark coat carries none of the bold black stripes of a tiger",
            "a photo of a bear, with a long straight snout instead of the short broad muzzle of a tiger",
            "a photo of a bear, whose shaggy coarse fur looks nothing like the sleek orange pelt of a tiger",
        ],
        ("bear", "wolf"): [
            "a photo of a bear, with a massive heavy body many times the bulk of a slender wolf",
            "a photo of a bear, showing small rounded ears rather than the sharp upright ears of a wolf",
            "a photo of a bear, whose stubby tail is nothing like the long bushy tail of a wolf",
        ],
        ("leopard", "bear"): [
            "a photo of a leopard, its golden coat covered in dark rosette spots that are absent on a bear",
            "a photo of a leopard, with a lithe muscular body much slimmer than a heavy bear",
            "a photo of a leopard, whose long tail trails far behind it unlike the stub of a bear",
        ],
        ("leopard", "lion"): [
            "a photo of a leopard, patterned with dark rosettes across a coat that a lion keeps plain",
            "a photo of a leopard, with a clean maneless head unlike the shaggy ruff of a lion",
            "a photo of a leopard, a compact cat noticeably smaller and lower slung than a lion",
        ],
        ("leopard", "tiger"): [
            "a photo of a leopard, marked with clustered rosette spots rather than the long stripes of a tiger",
            "a photo of a leopard, whose pale golden coat lacks the deep burnt orange tone of a tiger",
            "a photo of a leopard, a smaller slender cat beside the massive build of a tiger",
        ],
        ("leopard", "wolf"): [
            "a photo of a leopard, with a rounded feline face unlike the long canine muzzle of a wolf",
            "a photo of a leopard, its spotted golden coat set apart from the plain gray fur of a wolf",
            "a photo of a leopard, moving on soft padded paws with retractable claws no wolf has",
        ],
        ("lion", "bear"): [
            "a photo of a lion, the male framed by a thick shaggy mane that no bear grows",
            "a photo of a lion, with a short sandy coat instead of the long shaggy fur of a bear",
            "a photo of a lion, whose long tail ends in a dark tuft unlike the stub of a bear",
        ],
        ("lion", "leopard"): [
            "a photo of a lion, wearing a plain tawny coat free of the rosette spots of a leopard",
            "a photo of a lion, its head circled by a heavy mane that a leopard never has",
            "a photo of a lion, a far larger and more powerfully built cat than a leopard",
        ],
        ("lion", "tiger"): [
            "a photo of a lion, whose even sandy coat carries none of the black stripes of a tiger",
            "a photo of a lion, with a dense mane framing the face unlike a bare headed tiger",
            "a photo of a lion, showing a dark tuft at the tail tip that is absent on a tiger",
        ],
        ("lion", "wolf"): [
            "a photo of a lion, a large cat with a broad rounded face unlike the pointed snout of a wolf",
            "a photo of a lion, whose mane and tawny coat look nothing like the gray fur of a wolf",
            "a photo of a lion, far heavier and more muscular than a lean long legged wolf",
        ],
        ("tiger", "bear"): [
            "a photo of a tiger, striped in bold black bands that never appear on a bear",
            "a photo of a tiger, with a burning orange coat instead of the plain brown fur of a bear",
            "a photo of a tiger, a long low slung cat rather than a tall hunched bear",
        ],
        ("tiger", "leopard"): [
            "a photo of a tiger, marked with long vertical stripes rather than the rosettes of a leopard",
            "a photo of a tiger, whose deep orange coat is far richer than the pale gold of a leopard",
            "a photo of a tiger, a massive heavy cat beside the slender frame of a leopard",
        ],
        ("tiger", "lion"): [
            "a photo of a tiger, banded with black stripes that a plain coated lion entirely lacks",
            "a photo of a tiger, with a bare maneless head unlike a male lion",
            "a photo of a tiger, showing white markings on the face and belly that are absent on a lion",
        ],
        ("tiger", "wolf"): [
            "a photo of a tiger, a striped orange cat utterly unlike a gray furred wolf",
            "a photo of a tiger, with a broad rounded feline face instead of the long muzzle of a wolf",
            "a photo of a tiger, vastly heavier and more powerfully built than a slender wolf",
        ],
        ("wolf", "bear"): [
            "a photo of a wolf, a lean long legged animal far lighter than a bulky bear",
            "a photo of a wolf, with sharp upright ears unlike the small rounded ears of a bear",
            "a photo of a wolf, carrying a long bushy tail that a short tailed bear lacks",
        ],
        ("wolf", "leopard"): [
            "a photo of a wolf, with a long dog like muzzle unlike the flat feline face of a leopard",
            "a photo of a wolf, its grizzled gray coat free of any of the rosettes of a leopard",
            "a photo of a wolf, running on blunt fixed claws that a leopard would keep sheathed",
        ],
        ("wolf", "lion"): [
            "a photo of a wolf, a canine with a narrow pointed face unlike a broad headed lion",
            "a photo of a wolf, whose gray coat and bushy tail replace the tawny mane of a lion",
            "a photo of a wolf, lighter and leggier than a heavily muscled lion",
        ],
        ("wolf", "tiger"): [
            "a photo of a wolf, its plain gray coat carrying none of the stripes of a tiger",
            "a photo of a wolf, a dog like animal with a long snout unlike the short muzzle of a tiger",
            "a photo of a wolf, far smaller and leaner than a heavyset tiger",
        ],
    },
    "miniimagenet": {
        ("Walker hound", "Ibizan hound"): [
            "a photo of a Walker hound, with long pendulous ears that hang down unlike the tall erect ears of an Ibizan hound",
            "a photo of a Walker hound, its white coat broken by a black saddle and tan patches, not the red and white of an Ibizan hound",
            "a photo of a Walker hound, a solidly built scent hound rather than the racy fine boned Ibizan hound",
        ],
        ("Walker hound", "Saluki"): [
            "a photo of a Walker hound, with a short close lying coat lacking the silky feathering of a Saluki",
            "a photo of a Walker hound, showing a broad blocky muzzle unlike the narrow tapered head of a Saluki",
            "a photo of a Walker hound, tricolored in black white and tan rather than the pale cream of a Saluki",
        ],
        ("Walker hound", "Gordon setter"): [
            "a photo of a Walker hound, whose short smooth coat has none of the long silky fringe of a Gordon setter",
            "a photo of a Walker hound, marked with large white areas that are absent on a coal black Gordon setter",
            "a photo of a Walker hound, a lighter framed scent hound beside the heavier boned Gordon setter",
        ],
        ("Walker hound", "golden retriever"): [
            "a photo of a Walker hound, patched in black white and tan rather than the solid gold of a golden retriever",
            "a photo of a Walker hound, with a short flat coat unlike the dense feathered fur of a golden retriever",
            "a photo of a Walker hound, leaner and more angular than a stocky golden retriever",
        ],
        ("Ibizan hound", "Walker hound"): [
            "a photo of an Ibizan hound, with huge upright pointed ears that a drop eared Walker hound lacks",
            "a photo of an Ibizan hound, colored red and white without the black saddle of a Walker hound",
            "a photo of an Ibizan hound, tall and finely boned beside a sturdier Walker hound",
        ],
        ("Ibizan hound", "Saluki"): [
            "a photo of an Ibizan hound, whose ears stand straight up unlike the feathered drop ears of a Saluki",
            "a photo of an Ibizan hound, with a short hard coat instead of the silky fringe of a Saluki",
            "a photo of an Ibizan hound, showing a pink nose and amber eyes unlike a dark nosed Saluki",
        ],
        ("Ibizan hound", "Gordon setter"): [
            "a photo of an Ibizan hound, with erect pointed ears unlike the long drop ears of a Gordon setter",
            "a photo of an Ibizan hound, coated in short red and white rather than the black and tan of a Gordon setter",
            "a photo of an Ibizan hound, lean and leggy beside a heavier boned Gordon setter",
        ],
        ("Ibizan hound", "golden retriever"): [
            "a photo of an Ibizan hound, its tall erect ears nothing like the folded ears of a golden retriever",
            "a photo of an Ibizan hound, with a thin short coat unlike the thick golden fur of a golden retriever",
            "a photo of an Ibizan hound, narrow and deep chested rather than broad bodied like a golden retriever",
        ],
        ("Saluki", "Walker hound"): [
            "a photo of a Saluki, with long silky feathering on the ears and tail that is absent on a Walker hound",
            "a photo of a Saluki, a slender deep chested sighthound unlike a stockier Walker hound",
            "a photo of a Saluki, pale cream or fawn rather than the black white and tan of a Walker hound",
        ],
        ("Saluki", "Ibizan hound"): [
            "a photo of a Saluki, whose long feathered ears hang down unlike the erect ears of an Ibizan hound",
            "a photo of a Saluki, with silky fringed fur instead of the short hard coat of an Ibizan hound",
            "a photo of a Saluki, showing a dark nose and dark eyes unlike a pink nosed Ibizan hound",
        ],
        ("Saluki", "Gordon setter"): [
            "a photo of a Saluki, an extremely slender sighthound beside a solid muscular Gordon setter",
            "a photo of a Saluki, pale cream or fawn rather than the glossy black and tan of a Gordon setter",
            "a photo of a Saluki, with a deeply tucked waist and narrow head unlike a Gordon setter",
        ],
        ("Saluki", "golden retriever"): [
            "a photo of a Saluki, whose narrow waist and deep chest contrast the blocky body of a golden retriever",
            "a photo of a Saluki, with sparse silky feathering unlike the thick water repellent coat of a golden retriever",
            "a photo of a Saluki, having a long tapered skull rather than the broad head of a golden retriever",
        ],
        ("Gordon setter", "Walker hound"): [
            "a photo of a Gordon setter, with a long silky feathered coat unlike the short fur of a Walker hound",
            "a photo of a Gordon setter, coal black with tan points and none of the white patches of a Walker hound",
            "a photo of a Gordon setter, a heavier built gundog beside a lighter framed Walker hound",
        ],
        ("Gordon setter", "Ibizan hound"): [
            "a photo of a Gordon setter, with long drop ears unlike the upright pointed ears of an Ibizan hound",
            "a photo of a Gordon setter, its glossy black and tan coat unlike the red and white of an Ibizan hound",
            "a photo of a Gordon setter, solidly muscled rather than lean and racy like an Ibizan hound",
        ],
        ("Gordon setter", "Saluki"): [
            "a photo of a Gordon setter, deep black with rich tan markings unlike a pale cream Saluki",
            "a photo of a Gordon setter, a sturdy muscular gundog beside a fragile looking Saluki",
            "a photo of a Gordon setter, with a broad squared muzzle unlike the narrow head of a Saluki",
        ],
        ("Gordon setter", "golden retriever"): [
            "a photo of a Gordon setter, black with rich tan markings rather than the uniform gold of a golden retriever",
            "a photo of a Gordon setter, whose longer leaner frame contrasts the stocky build of a golden retriever",
            "a photo of a Gordon setter, with a narrower head than a broad skulled golden retriever",
        ],
        ("golden retriever", "Walker hound"): [
            "a photo of a golden retriever, in a solid golden coat without the tricolor patches of a Walker hound",
            "a photo of a golden retriever, with thick feathered fur unlike the short smooth coat of a Walker hound",
            "a photo of a golden retriever, broad and sturdy beside a leaner Walker hound",
        ],
        ("golden retriever", "Ibizan hound"): [
            "a photo of a golden retriever, whose ears fold close to the head unlike the erect ears of an Ibizan hound",
            "a photo of a golden retriever, densely coated in gold rather than thin short red and white",
            "a photo of a golden retriever, heavy and broad chested unlike a slim racy Ibizan hound",
        ],
        ("golden retriever", "Saluki"): [
            "a photo of a golden retriever, with a broad blocky head unlike the narrow skull of a Saluki",
            "a photo of a golden retriever, thickly coated in gold rather than the sparse silky fur of a Saluki",
            "a photo of a golden retriever, solidly built without the deeply tucked waist of a Saluki",
        ],
        ("golden retriever", "Gordon setter"): [
            "a photo of a golden retriever, uniformly gold rather than black with tan like a Gordon setter",
            "a photo of a golden retriever, with a broader heavier head than a Gordon setter",
            "a photo of a golden retriever, stockier and shorter coupled than a longer framed Gordon setter",
        ],
    },
    "cub200": {
        ("California Gull", "Glaucous-winged Gull"): [
            "a photo of a California Gull, with black wingtips that a pale winged Glaucous-winged Gull never shows",
            "a photo of a California Gull, standing on greenish yellow legs rather than the pink legs of a Glaucous-winged Gull",
            "a photo of a California Gull, a medium sized gull much smaller than a bulky Glaucous-winged Gull",
        ],
        ("California Gull", "Heermann's Gull"): [
            "a photo of a California Gull, with white underparts unlike the sooty gray body of a Heermann's Gull",
            "a photo of a California Gull, its yellow bill unlike the bright coral red bill of a Heermann's Gull",
            "a photo of a California Gull, showing greenish legs rather than the black legs of a Heermann's Gull",
        ],
        ("California Gull", "Herring Gull"): [
            "a photo of a California Gull, with greenish yellow legs instead of the pink legs of a Herring Gull",
            "a photo of a California Gull, having a dark eye unlike the pale staring eye of a Herring Gull",
            "a photo of a California Gull, whose bill carries both a black mark and a red spot unlike a Herring Gull",
        ],
        ("California Gull", "Ring-billed Gull"): [
            "a photo of a California Gull, larger with a darker gray mantle than a Ring-billed Gull",
            "a photo of a California Gull, its bill marked with red as well as black rather than a clean black ring",
            "a photo of a California Gull, with a dark eye rather than the pale eye of a Ring-billed Gull",
        ],
        ("Glaucous-winged Gull", "California Gull"): [
            "a photo of a Glaucous-winged Gull, whose wingtips are pale gray without any of the black of a California Gull",
            "a photo of a Glaucous-winged Gull, standing on pink legs rather than the greenish legs of a California Gull",
            "a photo of a Glaucous-winged Gull, a heavy bulky gull far larger than a California Gull",
        ],
        ("Glaucous-winged Gull", "Heermann's Gull"): [
            "a photo of a Glaucous-winged Gull, pale gray and white unlike the dark sooty Heermann's Gull",
            "a photo of a Glaucous-winged Gull, with a yellow bill instead of the red black tipped bill of a Heermann's Gull",
            "a photo of a Glaucous-winged Gull, on pink legs rather than the black legs of a Heermann's Gull",
        ],
        ("Glaucous-winged Gull", "Herring Gull"): [
            "a photo of a Glaucous-winged Gull, whose wingtips match the mantle instead of the black tips of a Herring Gull",
            "a photo of a Glaucous-winged Gull, with a dark eye unlike the pale yellow eye of a Herring Gull",
            "a photo of a Glaucous-winged Gull, bulkier with a much heavier bill than a Herring Gull",
        ],
        ("Glaucous-winged Gull", "Ring-billed Gull"): [
            "a photo of a Glaucous-winged Gull, a very large gull dwarfing a small Ring-billed Gull",
            "a photo of a Glaucous-winged Gull, with pale gray wingtips unlike the black tips of a Ring-billed Gull",
            "a photo of a Glaucous-winged Gull, on pink legs rather than the bright yellow legs of a Ring-billed Gull",
        ],
        ("Heermann's Gull", "California Gull"): [
            "a photo of a Heermann's Gull, its body dark sooty gray rather than the white of a California Gull",
            "a photo of a Heermann's Gull, with a bright coral red bill unlike a yellow billed California Gull",
            "a photo of a Heermann's Gull, standing on black legs rather than greenish yellow ones",
        ],
        ("Heermann's Gull", "Glaucous-winged Gull"): [
            "a photo of a Heermann's Gull, uniformly dark gray unlike a very pale Glaucous-winged Gull",
            "a photo of a Heermann's Gull, with a red black tipped bill rather than a plain yellow bill",
            "a photo of a Heermann's Gull, a slim medium sized gull beside a massive Glaucous-winged Gull",
        ],
        ("Heermann's Gull", "Herring Gull"): [
            "a photo of a Heermann's Gull, with sooty gray underparts where a Herring Gull is clean white",
            "a photo of a Heermann's Gull, its coral red bill unlike the yellow bill of a Herring Gull",
            "a photo of a Heermann's Gull, on black legs instead of the pink legs of a Herring Gull",
        ],
        ("Heermann's Gull", "Ring-billed Gull"): [
            "a photo of a Heermann's Gull, dark gray bodied unlike a white breasted Ring-billed Gull",
            "a photo of a Heermann's Gull, with a red bill tipped in black rather than a black ringed yellow bill",
            "a photo of a Heermann's Gull, standing on black legs rather than bright yellow legs",
        ],
        ("Herring Gull", "California Gull"): [
            "a photo of a Herring Gull, standing on bright pink legs unlike a greenish legged California Gull",
            "a photo of a Herring Gull, with a pale staring yellow eye rather than the dark eye of a California Gull",
            "a photo of a Herring Gull, a large heavy gull noticeably bigger than a California Gull",
        ],
        ("Herring Gull", "Glaucous-winged Gull"): [
            "a photo of a Herring Gull, with black wingtips marked by white spots unlike pale gray wingtips",
            "a photo of a Herring Gull, its pale yellow eye unlike the dark eye of a Glaucous-winged Gull",
            "a photo of a Herring Gull, slimmer with a lighter bill than a Glaucous-winged Gull",
        ],
        ("Herring Gull", "Heermann's Gull"): [
            "a photo of a Herring Gull, clean white below unlike the sooty gray body of a Heermann's Gull",
            "a photo of a Herring Gull, with a yellow bill bearing a red spot rather than a coral red bill",
            "a photo of a Herring Gull, on pink legs instead of the black legs of a Heermann's Gull",
        ],
        ("Herring Gull", "Ring-billed Gull"): [
            "a photo of a Herring Gull, much larger and heavier billed than a Ring-billed Gull",
            "a photo of a Herring Gull, with pink legs rather than the bright yellow legs of a Ring-billed Gull",
            "a photo of a Herring Gull, its bill marked by a red spot and not by a black ring",
        ],
        ("Ring-billed Gull", "California Gull"): [
            "a photo of a Ring-billed Gull, with a crisp black ring around the bill and no red spot",
            "a photo of a Ring-billed Gull, standing on bright yellow legs unlike the greenish legs of a California Gull",
            "a photo of a Ring-billed Gull, smaller with a paler gray mantle than a California Gull",
        ],
        ("Ring-billed Gull", "Glaucous-winged Gull"): [
            "a photo of a Ring-billed Gull, a small slight gull beside a huge Glaucous-winged Gull",
            "a photo of a Ring-billed Gull, with black wingtips unlike the pale gray wingtips of a Glaucous-winged Gull",
            "a photo of a Ring-billed Gull, on bright yellow legs rather than pink legs",
        ],
        ("Ring-billed Gull", "Heermann's Gull"): [
            "a photo of a Ring-billed Gull, white headed and white breasted unlike a dark sooty Heermann's Gull",
            "a photo of a Ring-billed Gull, with a yellow bill ringed in black rather than a coral red bill",
            "a photo of a Ring-billed Gull, standing on yellow legs instead of black legs",
        ],
        ("Ring-billed Gull", "Herring Gull"): [
            "a photo of a Ring-billed Gull, notably smaller and more delicate than a Herring Gull",
            "a photo of a Ring-billed Gull, with yellow legs rather than the pink legs of a Herring Gull",
            "a photo of a Ring-billed Gull, its bill circled by a black ring without any red spot",
        ],
    },
}


def build_matrix(dataset):
    """Return M as a nested list of shape (n, n, k); M[i][i] is None."""
    classes = CLASSES[dataset]
    pairs = PROMPTS[dataset]
    return [
        [None if i == j else pairs[(ci, cj)] for j, cj in enumerate(classes)]
        for i, ci in enumerate(classes)
    ]


def validate():
    for dataset, classes in CLASSES.items():
        n = len(classes)
        pairs = PROMPTS[dataset]
        expected = {(a, b) for a in classes for b in classes if a != b}
        missing = expected - set(pairs)
        extra = set(pairs) - expected
        assert not missing, f"{dataset}: missing pairs {sorted(missing)}"
        assert not extra, f"{dataset}: unexpected pairs {sorted(extra)}"
        for pair, prompts in pairs.items():
            assert len(prompts) == K, f"{dataset} {pair}: {len(prompts)} prompts, expected {K}"
            assert len(set(prompts)) == K, f"{dataset} {pair}: duplicate prompts"
        M = build_matrix(dataset)
        filled = sum(1 for row in M for cell in row if cell is not None)
        print(f"{dataset:14s} n={n}  M shape=({n}, {n}, {K})  "
              f"directed pairs={filled}  prompts={filled * K}")


if __name__ == "__main__":
    validate()
    print()
    M = build_matrix("cub200")
    classes = CLASSES["cub200"]
    i, j = 4, 3  # Ring-billed Gull vs Herring Gull
    print(f"M[{i}][{j}]  ({classes[i]} discriminated against {classes[j]}):")
    for p in M[i][j]:
        print(f"  - {p}")
