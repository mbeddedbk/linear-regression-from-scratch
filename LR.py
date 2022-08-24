
class LinearRegression():
    def __init__(self, learning_rate = 0.000005, ePoch = 1000):
        ''' constructor '''
        # Denklem agırlıklarının baslangicta belirlenmesi
        self.m1 = 1
        self.m2 = 2
        self.b = 0
        
        # Gradient Descent icin gerekli olan 
        # ogrenme parametresi ve iterasyon sayisinin atanmasi
        self.learning_rate = learning_rate
        self.ePoch = ePoch

    def gradDescent(self, L, xi, yi, zi):
        ''' Gradient Descent algoritmasi '''
        m1_grad = 0
        m2_grad = 0
        b_grad = 0
        loss = 0

        n = len(xi)

        for i in range(n):
            # tahmin edilmis z degeri
            zPredic = self.m1 * xi[i] + self.m2 * yi[i] + self.b
            loss = (zPredic) - zi[i]
            
            #agirliklarin gradyaninin hesaplanmasi
            m1_grad += (2/n) * xi[i] * (loss)
            m2_grad += (2/n) * yi[i] * (loss)
            b_grad += (2/n) * (loss)
        
        #agirlik guncellemesi yapilmasi
        self.m1 -= m1_grad * L
        self.m2 -= m2_grad * L # Bu degerleri dondurup hesap islemini jup de yap
        self.b -= b_grad * L

    def fit(self, x_train, y_train, z_train): #points dedigi veri setinin tamami
        ''' ogrenme fonksiyonu '''

        weights = []

        for i in range(self.ePoch):
            
            self.gradDescent(self.learning_rate, x_train, y_train, z_train)      
            # Loss ve accuracy grafiklerinin cozunurlukleri 4 degerde biri cizdirilerek dusuruldu
            if (i % 4 == 0):
                prevWeights = [i, self.m1, self.m2, self.b]
                weights.append(prevWeights)
        
        #Agirliklar sonra loss ve accuracy grafikleri icin geri donuldu
        return weights

    def predict(self, x_test, y_test):
        ''' tahmin fonksiyonu '''
        z_predicted = []
        n = len(x_test)

        for i in range(n):
            
            #guncellenmis agırlıklara gore yeni z degerlerinin tahmini
            new_val = self.m1 * x_test[i] + self.m2 * y_test[i] + self.b
            z_predicted.append(new_val)

        return z_predicted 
