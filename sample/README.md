# sample/

```
reference.wav   speaker enrollment clip — 24 kHz mono, 3–10 s recommended
reference.txt   exact transcript of reference.wav (typed by the user in the PoC)
target.txt      text to synthesise in that voice
```

Placeholder content until a real recording is made:

```
reference.txt : 안녕하세요. 이것은 제 목소리를 등록하기 위한 테스트 음성입니다.
target.txt    : 오늘 회의를 시작하겠습니다.
```

`reference.wav` is the one binary this repo does commit (see `.gitignore`); keep it under
1 MB. Generated outputs go to `out/` and are not committed.
