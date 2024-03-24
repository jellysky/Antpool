import time
import hmac,hashlib
import ssl,requests
coin_type='BTC'#币种
sign_id='s*****r'# 子账号名
sign_key='6b2f*************************69f'#密钥
sign_SECRET='f7e********7f'#密码
html_poolstats='https://antpool.com/api/poolStats.htm'
html_balance='https://antpool.com/api/account.htm'
html_hashrate_user='https://antpool.com/api/hashrate.htm'
html_hashrate_miner='https://antpool.com/api/workers.htm'
html_payment='https://antpool.com/api/paymentHistory.htm'
def get_signature():#签名操作
	nonce=int(time.time()*1000)#毫秒时间戳
	msgs=sign_id+sign_key+str(nonce)
	ret=[]
	ret.append(hmac.new(sign_SECRET.encode(encoding="utf-8"), msg=msgs.encode(encoding="utf-8"), digestmod=hashlib.sha256).hexdigest().upper())#签名
	ret.append(nonce)#时间戳
	return ret
def get_messages(url):#POST
	api_sign=get_signature()
	post_data = {'key': sign_key, 'nonce': api_sign[1],'signature': api_sign[0],'coin':coin_type}#这里是POST参数根据接口自行更改
	request = requests.post(url, data=post_data)
	return(request.text)

def main():
	print(get_messages(html_hashrate_user))
main()