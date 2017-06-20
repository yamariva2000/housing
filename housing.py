from datetime import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
"""Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15'],
      dtype='object')"""


class data():
    def __init__(self):
        # self.centers = centers
        # self.xm = xm
        df = pd.read_csv('/home/kel/housing/kc_house_data.csv - kc_house_data.csv', parse_dates=True)

        df.date = df.date.apply(lambda x: datetime.strptime(x[:-7], "%Y%m%d"))
        df['age'] = df.date.dt.year - df.yr_built
        df['time'] = (df.date - min(df.date)).astype(int)

        cond = df.yr_renovated == 0

        df.yr_renovated[cond] = df.yr_built

        df['renovation_recency'] = df.date.dt.year - df.yr_renovated
        self.df = df

    def plots(self):
        df = self.df

        df['price_ft'] = df.price / df.sqft_living
        hists = ['waterfront', 'view']
        f = plt.figure(1, figsize=(20, 15))
        for i, h in enumerate(hists, start=1):
            ax = f.add_subplot(2, 1, i)
            sns.distplot(df['price_ft'][df[h] == 1],
                         label='average {} ${:,.0f}/sft'.format(h, df['price_ft'][df[h] == 1].mean()), ax=ax)

            sns.distplot(df['price_ft'][df[h] == 0],
                         label='average no {} ${:,.0f}/sft'.format(h, df['price_ft'][df[h] == 0].mean()), ax=ax)
            ax.set_title('{} $/sft'.format(h), fontsize=20)
            ax.legend(fontsize=15)


        numberfeatures = ['grade', 'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'renovation_recency']

        f = plt.figure(2, figsize=(20, 15))

        df = df
        for i, feature in enumerate(numberfeatures, start=1):
            ax = f.add_subplot(2, 3, i)
            ax.scatter(df[feature], df['price'])
            ax.set_title(feature, fontsize=20)
            # ax.legend(fontsize=15)

        plt.show()

    def cluster(self):
        mdf = self.model_data()

        y = mdf.pop('price')
        x = mdf
        xm = x.drop(['month'], axis=1)

        scaler = StandardScaler()
        scaler2 = StandardScaler()
        normx = scaler.fit_transform(xm)
        normy = scaler2.fit_transform(y)
        n_clusters = 10
        k = KMeans(n_clusters=n_clusters)
        predictions = k.fit_predict(normx, normy)
        y=pd.DataFrame(y,columns=['price'])

        y['group']=predictions.reshape(-1,1)
        price_groups = y.groupby('group')['price'].agg(['mean', 'median', 'count'])

        centers = scaler.inverse_transform(k.cluster_centers_)
        centers = pd.DataFrame(centers, columns=xm.columns)

        centers = pd.concat((centers, price_groups), axis=1)

        centers.sort_values(by=['median'], inplace=True)
        self.centers=centers

        xm = pd.concat([xm, y['group']], axis=1)
        self.xm=xmx
        return xm['group']

    def map(self):
        self.cluster()
        centers = self.centers

        xm = self.xm
        max_median = np.max(centers['median'].values)

        for i in centers.index:
            group = xm[xm.group == i]
            size = min(1000, len(group))

            group = group.sample(size)

            plt.scatter(group.long, group.lat,
                        label='n={:n}  ${:0f}'.format(centers.loc[i]['count'], centers.loc[i]['median']),
                        c=cm.Reds(centers.loc[i]['median'] / max_median))

        #plt.scatter(centers['long'],centers['lat'],label='Centers',marker='^',s=200)

        plt.xlabel('Longitude', size=25)
        plt.ylabel('Latitude', size=25)
        plt.legend(fontsize=12)
        plt.title('Seattle-Bellevue Real Estate', fontsize=30)
        plt.show()

    def map2(self):
        df=self.df.sample(1000)

        df['ages']=pd.qcut(df['age'],q=6,labels=range(6))


        for group in sorted(df.ages.unique()):

            mask=df.ages==group

            plt.scatter(df.long[mask], df.lat[mask],
                        label='{}'.format(group))

                        # c=cm.Reds(centers.loc[i]['median'] / max_median))


        plt.xlabel('Longitude', size=25)
        plt.ylabel('Latitude', size=25)
        plt.legend(fontsize=12)
        plt.title('Seattle-Bellevue Real Estate', fontsize=30)
        plt.show()



    def model_data(self):

        mdf = self.df
        # mdf.drop(['id','date','lat','long'],axis=1,inplace=True)
        mdf.drop(['id'], axis=1, inplace=True)
        mdf['month'] = mdf.date.dt.month
        mdf['grade']=mdf['grade'].apply(lambda x: np.exp(x))
        mdf.index = mdf.date
        # mdf.drop(['date','yr_renovated','yr_built','zipcode','sqft_above','floors','month','time','view','age','sqft_living15','sqft_basement','condition','bathrooms','bedrooms','renovation_recency','sqft_lot','sqft_lot15'],axis=1,inplace=True)
        mdf.drop(['date', 'yr_renovated', 'yr_built', 'zipcode'], axis=1, inplace=True)
        print(mdf)
        return mdf

    def same_sales(self):
        mdf = self.model_data()
        mdf['date'] = mdf.index

        agg = mdf.groupby(['long', 'lat'])['price'].count().reset_index()
        agg.rename(columns={'price': 'count'}, inplace=True)

        mask = agg['count'] > 1

        merge = pd.merge(mdf, agg[mask], on=['long', 'lat'])
        merge.sort_values(by=['long', 'lat', 'date'], inplace=True)

        print(merge[['long', 'lat', 'date', 'renovation_recency', 'price']])

    def estimator(self):

        # groups=self.cluster()

        mdf = self.model_data()

        y = mdf.pop('price')
        # y.index=groups
        x = mdf

        dt = datetime(2015, 1, 1)

        xtrain, xtest, ytrain, ytest = x[x.index < dt], x[x.index >= dt], y[y.index < dt], y[y.index >= dt]

        model = RandomForestRegressor(n_estimators=20, n_jobs=4, random_state=1)
        #model=LinearRegression()
        model.fit(xtrain, ytrain)

        print(model.score(xtest, ytest))
        #print(xtrain.columns[np.argsort(model.feature_importances_)[::-1]])

        ytest_predict = pd.Series(model.predict(xtest), index=ytest.index, name='ypredict')

        ycombined = pd.concat([ytest, ytest_predict], axis=1)

        ycombined.rename(columns={'price': 'actual_price'}, inplace=True)

        ycombined['resid'] = ycombined['ypredict'] - ycombined['actual_price']
        # ycombined['price_buckets']=pd.qcut(ycombined.actual_price,q=5)
        ycombined['price_buckets'] = pd.cut(ycombined.actual_price,
                                            bins=[0, 300000, 500000, 700000, 1000000, 6000000])

        percent = .1
        ycombined['within'] = ((ycombined.actual_price < ycombined.ypredict * (1 + percent)) & (
            ycombined.actual_price > ycombined.ypredict * (1 - percent)))

        print(ycombined.groupby('price_buckets')['within'].agg({'percent': lambda x: sum(x) / len(x), 'count': len}))
        print(ycombined.head())

        exit()
        sample = ycombined.sample(100)
        # sample.sort_index(inplace=True)
        # plt.plot(sample.index,sample.actual_price,label='actual')
        # plt.plot(sample.index,sample.ypredict,label='est')
        # plt.legend()
        sns.violinplot(ycombined.index, ycombined.resid, ls='', marker='o')
        plt.show()


d = data()
#d.plots()
d.map()
#d.estimator()
# d.same_sales()
