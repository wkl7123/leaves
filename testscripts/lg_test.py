import lightgbm as lgb

bst = lgb.Booster(model_file='../testdata/lg_model.txt')
indexes = bst.predict([[0,1,63733,1111,36112,0,0,1,1]], pred_leaf=True)
print(indexes)