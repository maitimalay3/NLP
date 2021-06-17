# Assignmnet
# Question-Answer Datasets for Chatbot Training
# Example data Set

id	qid1	qid2	question1	question2	is_duplicate
0	1	2	What is the step by step guide to invest in share market in india?	What is the step by step guide to invest in share market?	0
1	3	4	What is the story of Kohinoor (Koh-i-Noor) Diamond?	What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?	0
2	5	6	How can I increase the speed of my internet connection while using a VPN?	How can Internet speed be increased by hacking through DNS?	0
3	7	8	Why am I mentally very lonely? How can I solve it?	Find the remainder when [math]23^{24}[/math] is divided by 24,23?	0
4	9	10	Which one dissolve in water quikly sugar, salt, methane and carbon di oxide?	Which fish would survive in salt water?	0
5	11	12	Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?	I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?	1
6	13	14	Should I buy tiago?	What keeps childern active and far from phone and video games?	0
7	15	16	How can I be a good geologist?	What should I do to be a great geologist?	1
8	17	18	When do you use シ instead of し?	"When do you use ""&"" instead of ""and""?"	0
9	19	20	Motorola (company): Can I hack my Charter Motorolla DCX3400?	How do I hack Motorola DCX3400 for free internet?	0
10	21	22	Method to find separation of slits using fresnel biprism?	What are some of the things technicians can tell about the durability and reliability of Laptops and its components?	0
11	23	24	How do I read and find my YouTube comments?	How can I see all my Youtube comments?	1
12	25	26	What can make Physics easy to learn?	How can you make physics easy to learn?	1
13	27	28	What was your first sexual experience like?	What was your first sexual experience?	1

#Data Preperation Code

from torchtext.legacy.data import Field, BucketIterator

data = pd.read_csv("quora_duplicate_questions.tsv",encoding = "ISO-8859-1", engine='python',delimiter="\t")
print(len(data))
data_new = data.dropna().reset_index().copy()
print((data_new.shape[0]))

Tweet = torchtext.legacy.data.Field(sequential = True, tokenize = 'spacy', batch_first =True, include_lengths=True)
Label = torchtext.legacy.data.LabelField(tokenize ='spacy', is_target=True, batch_first =True, sequential =False)

torchdata = [torchtext.legacy.data.Example.fromlist([data_new.question1[i],data_new.question2[i]], fields) for i in range(data_new.shape[0])]
fields = [('src', Tweet), ('trg', Tweet)]
FinalDataset = torchtext.legacy.data.Dataset(torchdata, fields)

(train, valid) = FinalDataset.split(split_ratio=[70, 30], random_state = random.seed(SEED))
train_iterator, valid_iterator = torchtext.legacy.data.BucketIterator.splits((train, valid), batch_size = 1, 
                                                            sort_key = lambda x: len(x.src),
                                                            sort_within_batch=True, device = device)

