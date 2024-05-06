from flask import jsonify


def success(params):
    return jsonify({
        'code': 200,
        'msg': 'OK',
        'data': params
    })


def fail(msg):
    return jsonify({
        'code': 0,
        'msg': msg
    })
