# V3 원문 질문·개체방향 정렬 — 별도 한정 학습

V2의 정상문장80/80과별개로미관측pair의긍정관계0/22가틀렸으므로, 단순같은loss의연장이아닌새질문정보reader/보조목적을검증한다. Root직접/noagents. 이전모든terminal/cap/checkpoint는닫히고불변이다.

신경망reader는원문에서계산된frozenbyte context만읽고4개의학습query로질문주어A–H/목적어A–H/greater-less-unknown/한영을예측한다. 네soft확률분포를학습embedding으로만들어문맥decoder와동결VP제안에전달한다. Decoder는이예측factor와VP만보며직접정답역할을주지않는다. 학습손실은ARbyteCE+.5*(주어CE+목적어CE+관계CE+.25언어CE)다. Offline target작성regex는train/scoringmetadata만생성하고모델추론경로에서금지한다. Goldfeature유입·미래byte누설·reader/decodergradient·zero-VPlatent불변성과실제planschema/전체OSowner경로를CPU검사한다.

부모는V2fullstate SHAb843d90b71f69419402fe3a6064522224c85058ab046b6f6fd1536348ca7d063. VPcacheSHA c0fe127015ddd17319e46d352be526264ffec0693990679585916051bca3fce9를그대로자료로재사용하며새VPencode/denoise는0이다. 이는옛미사용예산재사용이아니다. Decoder는부모를warmstart/reader는새학습/freshAdamW,stagecounter0이다. 부모Adam·기존가중치·모든raw를보존한다.

Prospective512attempt/최대512success,기존1024pool의고정4순열/4096rowexposures/seed30/batch8,새owner900초reserve90/output8GiB,actualreader+decoder2500호출(최대2254예상)이다. 0/128/256/384/512fullstate를검증저장하고중단시현재fullstate,회전삭제없음. LR3e-4/cosine512/clip1/AMP1024·3skip중단·수치/동결/hash/시간/예산/process/disk중단을고정한다. C36GiBstart/GPU12GiB/상시C20GiB+2GiB임시여유를지킨다. Root검토/실제preflight/독립validate/새SHA와고정clock을한번봉인한뒤ONEhiddenownedlaunch한다. 한도초과·자동재시작·oldterminal연장은없다.

두dev별exact75%이상,긍정문장exact60%이상,wellformed90%이상을새개발목표로둔다. Unknown상수기준선31.25%와head별주어·목적어·관계·언어정확도를따로본다. Zero-VP통제로learnedreader부분과확산의기여를구분하며어느결과도버리지않는다. 같은이름/문장중복·상관개발자료·좁은합성task의제약을명시한다. 일반LM/NER/vNext전체완성이나독립diffusion추론주장을하지않는다. E2final·원래typed학습은접근/재개하지않는다.
