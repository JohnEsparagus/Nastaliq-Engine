So Now I am sorucing the data for this project
we are going to use news soruces of sentences.

I am thinking of...
80% training data roughly
and the rest 20% or so would be testing data, 

we have 70k sentences so ill do 6:1
and maybe some more for some other testing. WE will see..

From my other project i alr know normalisation will be annoying especailly for urdu.
so many similar letters

I am gonna ask AI to analyse it
AI came back with some bad news
There is Arabic words in our urdu, well this is expected.
To be honest, that can be easily fixed in my opinion.

The worst thing is that a file was corrupted so lets see if we can fix that

Doing research online it seems that we should use a CRNN or CTC, if we want fast perofrmance but a vision transformer will always natively outperform. This is a low resources language unfortunately. I hope that will change soon. 

Some background on this, there isnt much literature in pakistan in urdu, I mean there is obviously a lot lot lot of rioch amazing poetry from the likes of Philospher Poet of the east Allama iqbal but ...
People type and communicate in romanized urdu and english. Pakistanis do not type in urdu so this is just some background , although there are 250 million speakers of urdu.


ok so now we are rendering text , 64px seems good.
We must account for height, i noticed i tried to render using bbox to calcualte the boundaries but it seemed a bit too cut off.




We implemented reverse_label becausei mean from what i know a crnn just takes from left to rigght, its easier that way so if we scan the ocr and output text such that labels are reversed, what labels areare the end result so the worlds it guesses, if that is reversed, then we would have no problem because left to right and reversed labels just means its readin git back wards...

its kjust that numbers in urdu are left to right, i struggled with this reading in urdu once so....
this is a proble rigght.

as a result I just reverse the string and then reverse the digits i guess. again....


## A5: height, stride and direction

The CNN cuts the line image into vertical strips, one every STRIDE pixels. Each strip (frame) can output at most one letter, and a doubled letter like کک needs a blank between the two, so it costs 3 frames not 2. If a line has fewer frames than its label needs, CTC literally can't output it.

So I measured it on 2000 real corpus lines. Each cell is the % of lines that fit, then how tight the worst line is (frames it gets / frames it needs, 1.00 = no room at all):

```
height  stride 2       stride 4       stride 8       stride 16
48      100.0% 2.01    100.0% 1.00      0.0% 0.50      0.0% 0.24
64      100.0% 2.68    100.0% 1.33      0.1% 0.67      0.0% 0.32
80      100.0% 3.36    100.0% 1.68     52.0% 0.83      0.0% 0.41
96      100.0% 4.03    100.0% 2.01    100.0% 1.00      0.0% 0.50
```

I picked **height 64, stride 4**.
- Every line fits, and the tightest one still has a third more frames than it needs. Augmentation shrinks lines a bit and real photos won't be the same scale, so I want that slack.
- 48 / 4 fits too but at exactly 1.00, zero margin. 96 / 8 is the same story and costs more compute because the image is taller.
- Stride 8 at 64 is a cliff: basically nothing fits. That's Nastaliq. Letters in a word stack diagonally, so several letters share the same horizontal pixels. Latin text would be fine at stride 8, Urdu isn't.

**Direction: reverse_label.** The CNN reads left to right, Urdu starts on the right, so I keep the image as it looks on the page and reverse the label instead. Images stay readable when I'm debugging, which flipping would ruin.

The catch was numbers. They're written left to right inside a right-to-left line, so `سال 2020 میں` on the image reads (from the left) ں ی م, then 2 0 2 0, then ل ا س. Reversing the whole string would give 0202 and train the model to read numbers backwards. That's 6,336 lines, 9% of the corpus.

Instead of reversing digits by hand I used the Unicode bidi algorithm (python-bidi), the same thing the renderer uses to lay out the line, so the label order matches the image by construction. One extra thing turned up: the renderer mirrors brackets in RTL text (it draws `(` where you'd expect `)`) and python-bidi doesn't, so I swap them after reordering. Checked against the actual rendered glyphs.

Going back from the model's output to normal text round-trips on 67,565 of 67,573 lines. The 8 that don't are weird ones like `1-2-3`. Doesn't affect training, only how those predictions would display.

HEIGHT now lives only in sequence.py, and render.py reads it from there. With that fixed I cached a clean render of every line (572 MB, about 6 min on all cores), so training only has to pay for augmentation.


squeeze(2)                (2, 512, 100)   drop H = 1 -> (N, C, T)
permute(2, 0, 1)          (100, 2, 512)   (T, N, C) for the LSTM
lstm                      (100, 2, 512)   2 directions x hidden
fc                        (100, 2, 75)   one score per character class

(2, 1, 64, 400) -> (100, 2, 75)
