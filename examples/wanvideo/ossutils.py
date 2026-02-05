import os
import os.path as osp
import oss2 as oss


bucket_fmt = 'oss://{}?endpoint={}&accessKeyID={}&accessKeySecret={}'

vgd_bucket = bucket_fmt.format('sora-data', 'oss-cn-wulanchabu-internal.aliyuncs.com', os.getenv('OSS_AK'), os.getenv('OSS_SK'))


def parse_oss_url(path):
    if path.startswith('oss://'):
        path = path[len('oss://'):]
    
    # configs
    configs = {
        'endpoint': os.getenv('OSS_ENDPOINT', None),
        'accessKeyID': os.getenv('OSS_ACCESS_KEY_ID', None),
        'accessKeySecret': os.getenv('OSS_ACCESS_KEY_SECRET', None),
        'securityToken': os.getenv('OSS_SECURITY_TOKEN', None)}
    bucket, path = path.split('/', maxsplit=1)
    if '?' in bucket:
        bucket, config = bucket.split('?', maxsplit=1)
        for pair in config.split('&'):
            k, v = pair.split('=', maxsplit=1)
            configs[k] = v
    
    # session
    session = parse_oss_url._sessions.setdefault(
        f'{bucket}@{os.getpid()}',
        oss.Session())
    
    # bucket
    bucket = oss.Bucket(
        auth=oss.Auth(configs['accessKeyID'], configs['accessKeySecret']),
        endpoint=configs['endpoint'],
        bucket_name=bucket,
        session=session)
    return bucket, path


parse_oss_url._sessions = {}

def parse_bucket(url):
    return parse_oss_url(osp.join(url, '_placeholder'))[0]

def read(filename, mode='r', retry=5):
    assert mode in ['r', 'rb']
    exception = None
    for _ in range(retry):
        try:
            if filename.startswith('oss://'):
                bucket, path = parse_oss_url(filename)
                content = bucket.get_object(path).read()
                if mode == 'r':
                    content = content.decode('utf-8')
            elif filename.startswith('http'):
                content = requests.get(filename).content
                if mode == 'r':
                    content = content.decode('utf-8')
            else:
                with open(filename, mode=mode) as f:
                    content = f.read()
            return content
        except Exception as e:
            exception = e
            continue
    else:
        raise exception