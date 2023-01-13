import math
import mesa
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Stemmer(mesa.Agent):
    '''
    Stemmer agent class

    hierin zit informatie over de stemmer, zoals zijn politieke positie, hoe kritisch die is en ook functies om
    de stem te bepalen.
    '''
    def __init__(self, unique_id, x, y, model):
        super().__init__(unique_id, model)
        self.x = x
        self.y = y
        self.vote = -1

        n = 99
        res = -1
        ballot = []
        for i in range(len(self.model.partijen)):
            p = self.model.partijen[i]
            new = math.sqrt((self.x - p[0])**2 + (self.y - p[1])**2)
            if new < n:
                n = new
                ballot.append(res)
                res = i
            else:
                ballot.append(i)

        ballot = ballot[1:]
        ballot.append(res)

        self.ballot = ballot
        self.c = random.randint(1, len(self.ballot) - 1) #hoe kritisch iemand is; wordt gebruikt in approval_vote

    def plurality_vote(self):
        '''
        stemt op favoriete partij
        als eer meerdere jaren gerunt worden kan er tactisch gestemd worden:
        wanneer de winner of de runner-up niet de favoriet is, is er een kans dat de stemmer besluit op de 'beste' van
        de twee te stemmen, iets wat je vaak ziet in bijvoorbeeld verkiezingen in de VS.
        hoe groter het aandeel stemmen van de top 2 partijen is, hoe eerder iemand tactisch zal stemmen
        '''
        res = self.ballot[len(self.ballot)-1]

        if (not self.model.data.empty):
            last_winnaar = int(self.model.winnaar)
            last_runnerup = int(self.model.data['vote'].value_counts().index[1])

        if (not self.model.data.empty) and (res != last_winnaar) and (res != last_runnerup):
            p = (self.model.data['vote'].value_counts().iloc[0] + self.model.data['vote'].value_counts().iloc[1]) / self.model.num_agents

            if random.random() < p: #tactisch stemmen:
                p1 = self.model.partijen[last_winnaar]
                p2 = self.model.partijen[last_runnerup]
                if math.sqrt((self.x - p1[0])**2 + (self.y - p1[1])**2) < \
                    math.sqrt((self.x - p2[0])**2 + (self.y - p2[1])**2):
                    self.vote = last_winnaar
                else: self.vote = last_runnerup
            else:
                self.vote = res
        else:
            self.vote = res

    def runoff_vote(self):
        '''
        stemt gewoon de partijen op volgorde van favoriet. Omdat je stem al op volgorde is is er niet veel incentive
        om tactisch te stemmen
        '''
        self.vote = self.ballot

    def approval_vote(self):
        '''
        stemt op het x aantal favoriete partijen; tactisch stemmen kan gebeuren wanneer de agent door heeft dat diens
        partij een grotere kans op winnen heeft wanneer er partijen van de stem weggelaten worden.
        '''

        self.vote = self.ballot[(len(self.ballot) - self.c):len(self.ballot)]
        if not self.model.data.empty and \
            (self.ballot[len(self.ballot)-1] != self.model.winnaar) and \
            (random.random() < .5):
            pos = self.ballot.index(self.model.winnaar)
            if (len(self.ballot) - self.c) <= pos:
                self.c -= (pos - (len(self.ballot) - self.c) + 1)
                self.vote = self.ballot[(len(self.ballot) - self.c):len(self.ballot)]


        # c = 2
        # len - c = 3
        #[ 0 , 1 , 2 , 3 , 4 ]
        #      w   c

    def step(self):
        if(self.model.systeem == 'p'):
            self.plurality_vote()
        elif(self.model.systeem == 'r'):
            self.runoff_vote()
        elif(self.model.systeem == 'a'):
            self.approval_vote()
        else:
            self.vote = -1


class KiesModel(mesa.Model):
    '''
    Model class
    '''
    def __init__(self, n_stem, n_partij, systeem):
        self.num_agents = n_stem
        self.systeem = systeem
        self.partijen = []
        self.winnaar = -1
        self.schedule = mesa.time.RandomActivation(self)
        # self.space = mesa.space.ContinuousSpace(1,1,False)

        for i in range(n_partij):
            self.partijen.append((random.random(), random.random()))

        self.datacollector = mesa.DataCollector(
            agent_reporters={"x_position": "x", "y_position": "y", "vote": "vote"})

        self.data = pd.DataFrame(
            {"x_position": pd.Series(dtype=float), "y_position": pd.Series(dtype=float), "vote": pd.Series(dtype=int)})

        for i in range(n_stem):
            x = random.random()
            y = random.random()
            a = Stemmer(i, x, y, self)
            # self.space.place_agent(a, (x,y))
            self.schedule.add(a)

    def step(self):
        '''
        laat iedereen stemmen en voegt de stemmen toe aan data

        bepaald vervolgens de winnaar aan de hand van het gekozen kiesstelsel
        '''
        self.schedule.step()
        self.datacollector.collect(self)

        # self.data = pd.DataFrame(
        #     {"x_position": pd.Series(dtype=float), "y_position": pd.Series(dtype=float), "vote": pd.Series(dtype=int)})

        # for i in self.schedule.agents:
        #     self.data.loc[-1] = [i.pos[0],i.pos[1],i.vote]
        #     self.data.index = self.data.index+1

        self.data = self.datacollector.get_agent_vars_dataframe().tail(self.num_agents)

        if self.systeem == 'p':
            self.winnaar = self.data['vote'].mode()[0] #meest voorkomende stem is de winnaar
        elif self.systeem == 'r':
            self.winnaar = self.runoff_vote()
        elif self.systeem == 'a':
            self.winnaar = self.approval_vote()
        else:
            self.winnaar = -1

    def approval_vote(self):
        '''
        de winnaar is de partij met de meeste stemmen
        '''
        s = pd.Series(np.hstack(self.data['vote']))
        return s.value_counts().index[0]

    def runoff_vote(self):
        '''
        bepaald de winnaar door te kijken of er een partij is met meer dan 50% nr1 stemmen, als dit niet het geval is
        wordt de partij met de minste n1 stemmen geelimineerd; dit wordt herhaald tot er een winnende partij gevonden is
        '''
        def shift_row(a):
            if a.iloc[-1] == -1:
                n = 1
                for j in range(len(a)-1):
                    try:
                        if a.iloc[-(2+j)] == -1:
                            n+=1
                        else:
                            break
                    except TypeError:
                        break

                return a.shift(periods=n)
            return a

        df = pd.DataFrame(np.vstack(self.data['vote']))

        for i in range(len(self.partijen)-1):
            print(df[len(self.partijen) - 1].value_counts())
            if df[len(self.partijen)-1].value_counts().iloc[0] >= (self.num_agents/2):
                return df[len(self.partijen)-1].value_counts().index[0]
            else:
                p = df[len(self.partijen)-1].value_counts().index
                verliezer = p[len(p)-1]
                df.replace(to_replace=verliezer, value=-1, inplace=True)
                df = df.apply(lambda a: shift_row(a), axis=1).astype('Int64')

        print(df[len(self.partijen) - 1].value_counts())
        # print('tiee')
        return -1

    def condorcet(self):
        '''
        vindt de condorcet winnaar:
        elke partij doet een pairwise runoff tegen elke andere partij, de partij die van elke partij wint is de
        condorcet winnaar. wanneer er geen condorcet winnaar is wordt er -1 gereturnt
        '''

        #df = self.datacollector.get_agent_vars_dataframe()
        df = self.data
        p1 = self.partijen[0]
        n = 0
        for i in range(len(self.partijen)-1):
            # percentage = round(i / (len(self.partijen)-1) * 100, 2)
            # print('Progress: [{}{}{}]  {}%'.format(('=' * int(percentage // 10)), ('>' if percentage < 100 else ''),
            #                                             ('.' * int(10 - (((percentage) // 10)) - 1)), percentage))

            p2 = self.partijen[i+1]
            df['n'] = df.apply(lambda a: self.pairwise_runoff(p1,p2,(a['x_position'], a['y_position']),n,i+1), axis=1)
            n = df['n'].mode()[0]
            #print(df['n'])
            p1 = self.partijen[n]

        if n!=0:
            for i in range(len(self.partijen)):
                if (i == n): break
                p2 = self.partijen[i]
                df['n'] = df.apply(lambda a:
                                   self.pairwise_runoff(p1,p2,(a['x_position'], a['y_position']),n,i+1), axis=1)
                if df['n'].mode()[0] != n:
                    print('tie')
                    return -1

        return n

    def pairwise_runoff(self, p1, p2, v, n1, n2):
        if((math.sqrt((p1[0] - v[0])**2 + (p1[1] - v[1])**2) - math.sqrt((p2[0] - v[0])**2 + (p2[1] - v[1])**2)) > 0):
            return n2
        else:
            return n1


def plurality_printres(n_jaar, n_stem, n_partij):
    newmodel = KiesModel(n_stem, n_partij, 'p')
    newmodel.step()
    df = newmodel.data
    print('partijen: ')
    for i in newmodel.partijen:
        print('\t' + str(i))
    print('\nn stemmers: \t\t' + str(newmodel.num_agents))
    print('condorcet winnaar:\t' + str(newmodel.condorcet()))
    print('winnaar:\t\t\t' + str(df['vote'].value_counts().index[0]))
    print('stemmen:\n' + str(df['vote'].value_counts()))

    for i in range(n_jaar):
        newmodel.step()
        df = newmodel.data
        print('\nwinnaar:\t\t\t' + str(df['vote'].value_counts().index[0]))
        print('stemmen:\n' + str(df['vote'].value_counts()))

    return (newmodel.winnaar, newmodel.condorcet())

def approval_printres(n_jaar, n_stem, n_partij):
    newmodel = KiesModel(n_stem, n_partij, 'a')
    newmodel.step()
    # df = newmodel.data
    print('partijen: ')
    for i in newmodel.partijen:
        print('\t' + str(i))
    print('\nn stemmers: \t\t' + str(newmodel.num_agents))
    print('condorcet winnaar:\t' + str(newmodel.condorcet()))
    print('winnaar:\t\t\t' + str(newmodel.winnaar))
    s = pd.Series(np.hstack(newmodel.data['vote']))
    print('stemmen:\n' + str(s.value_counts()))

    for i in range(n_jaar):
        newmodel.step()
        print('winnaar:\t\t\t' + str(newmodel.winnaar))
        s = pd.Series(np.hstack(newmodel.data['vote']))
        print('stemmen:\n' + str(s.value_counts()))

    return (newmodel.winnaar, newmodel.condorcet())

def runoff_printres(n_jaar, n_stem, n_partij):
    newmodel = KiesModel(n_stem, n_partij, 'r')
    newmodel.step()
    # df = newmodel.data
    print('partijen: ')
    for i in newmodel.partijen:
        print('\t' + str(i))
    print('\nn stemmers: \t\t' + str(newmodel.num_agents))
    print('condorcet winnaar:\t' + str(newmodel.condorcet()))
    print('winnaar:\t\t\t' + str(newmodel.winnaar))
    s = pd.Series(np.hstack(newmodel.data['vote']))
    print('stemmen:\n' + str(s.value_counts()))

    for i in range(n_jaar):
        newmodel.step()
        print('winnaar:\t\t\t' + str(newmodel.winnaar))
        s = pd.Series(np.hstack(newmodel.data['vote']))
        print('stemmen:\n' + str(s.value_counts()))

    return (newmodel.winnaar, newmodel.condorcet())


def plurality(n_jaar, n_stem, n_partij):
    newmodel = KiesModel(n_stem, n_partij, 'p')
    newmodel.step()
    c = newmodel.condorcet()
    for i in range(n_jaar):
        newmodel.step()

    return (int(newmodel.winnaar), int(c))


def instant_runoff(n_jaar, n_stem, n_partij):
    newmodel = KiesModel(n_stem, n_partij, 'r')
    newmodel.step()
    c = newmodel.condorcet()
    return (int(newmodel.winnaar), int(c))


def approval(n_jaar, n_stem, n_partij):
    newmodel = KiesModel(n_stem, n_partij, 'a')
    newmodel.step()
    c = newmodel.condorcet()
    return (int(newmodel.winnaar), int(c))


# approval_printres(5, 10000, 5)
# plurality_printres(5, 10000, 5)
runoff_printres(0, 100000, 5)


# n_year = 5
# n_stem = 1000
# parties = 3
# runs = 100
#
# p1 = np.arange(n_year)
# p2 = np.arange(n_year)
# p3 = np.arange(n_year)
#
# for j in range(runs):
#     newmodel = KiesModel(n_stem, parties, 'p')
#     for i in range(n_year):
#         newmodel.step()
#         list1 = newmodel.data['vote'].value_counts()
#         p1[i] += (list1.iloc[0])/n_stem*100
#         p2[i] += (list1.iloc[1])/n_stem*100
#         if(len(list1)==3):
#             p3[i] += (list1.iloc[2])/n_stem*100
#     p1 = p1/runs
#     p2 = p2/runs
#     p3 = p3/runs
#     print(j)
#
#
# labels = list(np.arange(n_year))
#
# x = np.arange(len(labels))
# width = 0.25
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width, p1, width, label='partij 1')
# rects2 = ax.bar(x,         p2, width, label='partij 2')
# rects3 = ax.bar(x + width, p3, width, label='partij 3')
#
# ax.set_xlabel('n election')
# ax.set_ylabel('% of votes')
# ax.set_xticks(x, labels)
# ax.legend()
#
# fig.tight_layout()
#
# plt.show()


# runs = 1000
# voters = 1000
# parties = 6
#
# y_a = []
# for i in range(3,parties):
#     n = 0
#     for j in range(runs):
#         res = approval(5, voters, i)
#         if(res[0] == res[1]):
#             n+=1
#     print((n/runs) * 100)
#     y_a.append((n/runs) * 100)
# print(y_a)

# y_p = []
# for i in range(3,parties):
#     print(i)
#     n = 0
#     for j in range(runs):
#         res = plurality(5, voters, i)
#         if(res[0] == res[1]):
#             n+=1
#     print((n/runs) * 100)
#     y_p.append((n/runs) * 100)
# print('p')
#
# y_r = []
# for i in range(3,parties):
#     n = 0
#     for j in range(runs):
#         res = instant_runoff(0, voters, i)
#         if(res[0] == res[1]):
#             n+=1
#     print((n/runs) * 100)
#     y_r.append((n/runs) * 100)
# print('r')

# labels = list(np.arange(parties))[3:]
#
# x = np.arange(len(labels))
# width = 0.25
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width, y_p, width, label='plurality')
# rects2 = ax.bar(x,         y_r, width, label='instant runoff')
# rects3 = ax.bar(x + width, y_a, width, label='approval')
#
# ax.set_xlabel('n parties')
# ax.set_ylabel('% of elections resulting in condorcet winner')
# ax.set_xticks(x, labels)
# ax.legend()
#
# fig.tight_layout()
#
# plt.show()

