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

monet_normal = [
    "painting of women working in the garden",
    "paintinf of rocks in the ocean",
    "a painting of a city",
    "a painting of a river",
    "painting of a person on a cliff",
    "a painting of a town",
    "a painting of a sunset",
    "a painting of mountains",
    "painting of flowers in a field",
    "a painting of a landscape",
    "two trees in a field in a painting",
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

dali_normal = [
    "painting of the persistence of memory",
    "painting of an elephant",
    "painting of soft construction with boiled beans",
    "painting of galatea of the spheres",
    "painting of the temptation of st. anthony",
    "painting of swans reflecting elephants",
    "painting of slave market with the disappearing bust",
    "painting of a meditative rose",
    "painting of melting",
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

mann_normal = [
    "painting of a view of a city skyline at sunset, with a warm glow spreading across the sky and the buildings below",
    "painting of an urban scene of a group of people gathered on a street corner, captured in a moment of quiet reflection",
    "painting of a surreal composition of floating objects, with a dreamlike quality to the light and color",
    "painting of a view of a city street at night, with the glow of streetlights and neon signs casting colorful reflections on the wet pavement",
    "painting of a moody, atmospheric scene of a dark alleyway, with a hint of warm light glowing in the distance",
    "painting of an urban scene of a group of people walking through a park, captured in a moment of movement and energy",
    "painting of a landscape of a forest, with dappled sunlight filtering through the leaves and a sense of stillness and peace",
    "painting of a surreal composition of architectural details and organic forms, with a sense of tension and unease in the composition",
    "painting of an abstract composition of geometric shapes and intricate patterns, with a vibrant use of color and light",
    "painting of a painting of a bustling city at night, captured in the rain-soaked streets and neon lights",
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

greg_normal = [
    "painting of a man riding a horse, dragon breathing fire",
    "painting of a dragon attacking a knight",
    "painting of  a demonic creature in the wood",
    "painting of a man in a forbidden city",
    "painting of a group of people on a dock",
    "painting of a king standing, with people around in a hall",
    "painting of two magical characters in space",
    "painting of a man with a fire in his hands",
    "painting of a woman sitting on a couch",
    "painting of a man with a sword standing on top of a pile of skulls",
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

van_gogh_normal = [
    "a painting of rocky ocean shore under the night sky",
    "A painting of lone figure contemplates on a cliff",
    "painting of majestic mountains take on an ethereal quality in sky",
    "Painting of two trees in a sunlit field",
    "Painting of a flower-filled field",
    "painting of a river on a warm sunset",
    "painting of olive trees",
    "painting of a field with mountains in the background",
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

picasso_normal = [
    "Painting of nude figures",
    "painting of a grieving woman",
    "painting of three dancers",
    "portrait of a girl in front of mirror",
    "painting of a bird",
    "painting of a blind musician",
    "painting of a room",
    "painting of an acrobat in performance",
]

style_templates = [
    "painting of olive trees in the style of {}",
    "painting of a tree in the style of {}",
    "the starry night painting in the style of {}",
    "painting of women working in the garden, in the style of {}",
    "a painting of a wheat field by {}",
    "painting of trees in bloom in the style of {}",
    "{} style painting of a tree",
    "painting of a wheat field in the style of {}",
    "{} style painting of a field with mountains in the background",
    "a painting of trees in bloom in the style of {}"
]

extensive_style_templates = [
    "painting of a {} in the style of {}",
    "painting of a {} by {}",
    "{} style painting of a {}",
]

normal_templates = [
    "painting of olive trees",
    "painting of a tree",
    "painting of women working in the garden",
    "a painting of a wheat field",
    "painting of trees in bloom",
    "painting of a field with mountains in the background",
    "a painting of trees in bloom",
]

extensive_normal_templates = [
    "painting of a {}",
    "a painting of a {}",
]

objects = ['tree', 'olive trees', 'field with mountains', 'wheat field',
           'women in the garden', 'night', 'dog', 'cat',
           'bicycle', 'oven', 'bowl', 'bottle',
           'apple', 'cup', 'bed', 'clock']

artists = ['van gogh', 'monet', 'leonardo da vinci', 'baroque','pablo picasso', 'salvador dali', 'michelangelo']

colors = ['blue', 'red', 'yellow', 'brown','black', 'pink', 'green']


