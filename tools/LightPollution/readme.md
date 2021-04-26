df = pd.read_csv(os.path.join(Folder_Path,'WDPA_ntl_per_buffer'+str(i)+'.csv'))
df = df.drop(columns=['system:index','.geo'])
basedf = basedf.merge(df, on='objectid',how='left',suffixes = ('','_'+str(i)))

<!-- opencv -->
convertedImage = cv2.cvtColor(image, cv2.COLOR_BayerGB2BGR)


<!-- sklearn -->
def scoreFor(csvname):
    df= pd.read_csv(csvname) 
    y = np.array(df[['Validation']])
    y_pred = np.array(df[['classification']])
    reports = classification_report(y, y_pred)
    confusion = confusion_matrix(y, y_pred, labels=None, sample_weight=None)
    score = accuracy_score(y, y_pred)
    print(confusion)
    print(reports)
    return score


from scipy.optimize import curve_fit
from sklearn import metrics
def func(x,a,b,c,d):
    return a + b*(1/(1 + np.exp(-c*(x-d))))
popt, pcov = curve_fit(func, x, y, bounds = param_bounds)

def writefile(filename,lines):
    f = open(filename,'a')
    for line in lines:
        f.writelines(line + '\n')

<!-- GDAL addID -->
def featureAddId(shp_filename):
    ds=ogr.Open(shp_filename,1)
    lyr=ds.GetLayer()
    print('Sample feature number is: ',lyr.GetFeatureCount())
    fieldDefn = ogr.FieldDefn('objectId', ogr.OFTInteger)
    lyr.CreateField(fieldDefn)
    Ids = 0
    for feat in lyr:
        Ids = Ids + 1
        print(Ids)
        feat.SetField('objectId', Ids)
        lyr.SetFeature(feat)
