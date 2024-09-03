# monet
monet_prompts = [
    "painting of women working in the garden in the style of {}",
    "rocks in the ocean, in the style of {}",
    "a painting of a city in the style of {}",
    "a painting of a river in the style of {}",
    "{} style painting of a person on a cliff",
    "a painting of a town, in the style of {}",
    "a painting of a sunset, in the style of {}",
    "a painting of mountains, in the style of {}",
    "{} style painting of flowers in a field",
    "a painting of a landscape in the style of {}",
    "two trees in a field, painting in the style of {}",
]

# salvador dali
dali_prompts = [
    "the persistence of memory painting in the style of {}",
    "the elephant painting in the style of {}",
    "soft construction with boiled beans painting in the style of {}",
    "galatea of the spheres painting in the style of {}",
    "the temptation of st. anthony painting in the style of {}",
    "swans reflecting elephants painting in the style of {}",
    "enigma of desire painting in the style of {}",
    "slave market with the disappearing bust of voltaire painting of {}",
    "the meditative rose painting in the style of {}",
    "melting watch painting in the style of {}",
]

# jeremy mann
mann_prompts = [
    "In the style of {}, a view of a city skyline at sunset, with a warm glow spreading across the sky and the buildings below",
    "In the style of {}, an urban scene of a group of people gathered on a street corner, captured in a moment of quiet reflection",
    "In the style of {}, a surreal composition of floating objects, with a dreamlike quality to the light and color",
    "In the style of {}, a view of a city street at night, with the glow of streetlights and neon signs casting colorful reflections on the wet pavement",
    "In the style of {}, a moody, atmospheric scene of a dark alleyway, with a hint of warm light glowing in the distance",
    "In the style of {}, an urban scene of a group of people walking through a park, captured in a moment of movement and energy",
    "In the style of {}, a landscape of a forest, with dappled sunlight filtering through the leaves and a sense of stillness and peace",
    "In the style of {}, a surreal composition of architectural details and organic forms, with a sense of tension and unease in the composition",
    "In the style of {}, an abstract composition of geometric shapes and intricate patterns, with a vibrant use of color and light",
    "In the style of {}, a painting of a bustling city at night, captured in the rain-soaked streets and neon lights",
]

# greg rutkowski
greg_prompts = [
    "a man riding a horse, dragon breathing fire, painted by {}",
    "a dragon attacking a knight in the style of {}",
    "a demonic creature in the wood, painting by {}",
    "a man in a forbidden city, in the style of {}",
    "painting of a group of people on a dock by {}",
    "a king standing, with people around in a hall, painted by {}",
    "two magical characters in space, painting by {}",
    "a man with a fire in his hands in the style of {}",
    "painting of a woman sitting on a couch by {}",
    "a man with a sword standing on top of a pile of skulls, in the style of {}",
]

# van gogh
van_gogh_prompts = [
    "a painting of rocky ocean shore under the luminous night sky in the style of {}",
    "A painting of lone figure contemplates on a cliff, surrounded by swirling skies in the style of {}",
    "Majestic mountains take on an ethereal quality in sky, painted by {}",
    "Two trees in a sunlit field, painted by {}",
    "A flower-filled field in the style of {}",
    "painting of a river on a warm sunset in the style of {}",
    "painting of olive trees in the style of {}",
    "painting of a field with mountains in the background in the style of {}",
]

# pablo picasso
picasso_prompts = [
    "Painting of nude figures by {}",
    "painting of a grieving woman in the style of {}",
    "painting of three dancers by {}",
    "portrait of a girl in front of mirror, painted by {}",
    "painting of a bird, in the style of {}",
    "painting of a blind musician, by {}",
    "painting of a room in the style of {}",
    "painting of an acrobat in performance, by {}",
]

test_prompts = {
    'monet': [x.format('monet') for x in monet_prompts],
    'salvador dali': [x.format('salvador dali') for x in dali_prompts],
    'jeremy mann': [x.format('jeremy mann') for x in mann_prompts],
    'greg rutkowski': [x.format('greg rutkowski') for x in greg_prompts],
    'van gogh': [x.format('van gogh') for x in van_gogh_prompts],
    'pablo picasso': [x.format('pablo picasso') for x in picasso_prompts],
}

