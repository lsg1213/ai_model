import json, argparse, os, pdb

def saveConfig(path, name, config):
    jsonpath = os.path.join(path, name)
    with open(jsonpath, 'w') as f:
        json.dump(config, f, sort_keys=True)

def loadConfig(path, name):
    if not (os.path.splitext(name)[-1] == '.json'):
        name += '.json'
    jsonpath = os.path.join(path, name)
    if os.path.exists(jsonpath): # name.json exists
        with open(jsonpath, 'r') as f:
            res = json.load(f)
        return res
    else:
        print(f"{jsonpath} config don't exists")
        raise ValueError()

def manageVersion(name):
    name = os.path.splitext(name)[0]
    if '_v(' in name:
        oldversion = int(name.split('_v(')[-1][:-1])
        newversion = str(oldversion + 1)
        name = name.replace(f'_v({str(oldversion)})', f'_v({newversion})')
    else:
        name += '_v(0)'
    return name + '.json'

def getRealName(jsonpath):
    name = os.path.basename(jsonpath)
    if '_v(' in name:
        name = name.split('_v(')[0]
    return name

def findDuplicateConfig(jsonpath, newconfig):
    from glob import glob
    name = getRealName(jsonpath)
    for i in glob(os.path.dirname(os.path.abspath(jsonpath)) + f'/{name}*.json'):
        _config = loadConfig(os.path.dirname(i), os.path.basename(i))
        if _config == newconfig:
            return i
    return False


def getConfig(name:str, 
              config:argparse.Namespace, 
              use_only_saved:bool, 
              path:str='./config', 
              savemode:str='b'):
    '''
    name: CONFIG.json name
    config: parsed config
    use_only_saved: if True, ignore your typed configuration
    path: CONFIG.json path
    savemode: (a: save overwrite, b: save version update, c: no save), default is b
    '''
    assert len(name) > 0, 'name must be typed'
    assert savemode in ['a','b','c'], 'save mode must be a, b, or c'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'configuration directory is made at {path}')
    if not (os.path.splitext(name)[-1] == '.json'):
        name += '.json'
    jsonpath = os.path.join(path, name)
    
    if os.path.exists(jsonpath): # 기존 config의 존재
        loaded_config = loadConfig(path, name)
        for key in vars(config): # 저장된 config에 현재 config 반영 (config 형식이 다를 경우 전제)
            if use_only_saved: # 저장된 config만 사용
                return argparse.Namespace(**loaded_config)
            loaded_config[key] = getattr(config, key)
    else: # 기존 config의 미존재
        if use_only_saved:
            raise ValueError('you have to load existing configuration')
        loaded_config = config

    # 기존 config와 비교해서 동일한 config가 있는 지 탐색
    dup = findDuplicateConfig(jsonpath, loaded_config)
    if dup:
        print(f'{dup} is the same with your configuration')
        raise ValueError()
    if savemode == 'b':
        name = manageVersion(name)
    if savemode != 'c':
        saveConfig(path, name, loaded_config)
    return argparse.Namespace(**loaded_config)
        

        
    


if __name__ == "__main__":
    import sys
    arg = argparse.ArgumentParser()
    arg.add_argument('--hi', type=str, default='bye')
    config = arg.parse_known_args(sys.argv[1:])[0]
    getConfig('b_v(0)', config, False)