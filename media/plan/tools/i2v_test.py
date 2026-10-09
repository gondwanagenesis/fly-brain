import os, json, base64, urllib.request, time, sys
KEY=os.environ['FAL_KEY']
img=sys.argv[1]; prompt=sys.argv[2]; dur=sys.argv[3]; out=sys.argv[4]
uri='data:image/png;base64,'+base64.b64encode(open(img,'rb').read()).decode()
body={"image_url":uri,"prompt":prompt,"duration":dur}
req=urllib.request.Request("https://fal.run/fal-ai/kling-video/v3/turbo/standard/image-to-video",data=json.dumps(body).encode(),headers={"Authorization":f"Key {KEY}","Content-Type":"application/json"})
t=time.time()
try:
    r=json.load(urllib.request.urlopen(req,timeout=900))
except urllib.error.HTTPError as e:
    print('HTTP',e.code,e.read()[:800]); sys.exit(1)
print('ok',round(time.time()-t),'s',json.dumps(r)[:300])
urllib.request.urlretrieve(r['video']['url'],out)
