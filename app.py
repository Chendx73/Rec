import json
from flask import request, Flask
from Recall.Hot_TopN import get_top_n
from utils.params import success, fail

app = Flask(__name__)


@app.route('/hotN', methods=['POST'])
def getTopN():
    data = json.loads(request.get_data())
    if data == {} or not isinstance(data['N'], int):
        return fail('数据格式不对')
    res = get_top_n(data['N'])
    return success(res)
