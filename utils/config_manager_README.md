# 사용법

import sys, argparse
known = sys.argv[:1]
args = argparse.ArgumentParser()
args.add_argument로 각종 argument 추가
(--name은 반드시 있어야 함, str type)
config = args.parse_known_args(known)[0]
config = getConfig(config.name, config, mode)


## mode:
- 'l': name과 동일한 config를 찾아서 가져와서 해당 config에 입력한 argument를 덮어 씌운다, 해당 이름의 name이 없으면 ValueError()
- 'o': 사용 불가
- 'lo' or 'ol': name과 동일한 config만 가져와서 gpus만 덮어씌우고 나머지 argument는 무시한다.
