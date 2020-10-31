我取了部分我觉得可能有关联的data，都是反映个人性格的
age -> 提取关键字得到数字
anagrams 1-4 -> 和模版的编辑距离
backcount 1-10 -> 360减3递减，正确答案是true，其他都是false
gender -> 映射成数字，1，2，3不变，4-16根据下表
        gender_dict = {4: {'agender'}, 5: {'androgyne', 'androgynous'}, 6: 'bigender', 
                    7: {'cis', 'cis female', 'cis woman', 'cis male', 'cis man', 'cisgender',
                        'cisgender female', 'cisgender male', 'cisgender man', 'cisgender woman'},
                    8: {'female to male', 'ftm'}, 9: {'gender fluid'}, 10: {'gender nonconforming'},
                    11: {'gender questioning', 'gender variant', 'genderqueer', 'intersex', 
                        'male to female', 'mtf', 'neither', 'neutrois', 'non-binary'}, 
                    12: {'pangender'}, 13: {'trans', 'trans female', 'trans male', 'trans man',
                        'trans person', 'trans woman', 'trans*', 'trans* female', 'trans* male',
                        'trans* man', 'trans* person', 'trans* woman', 'transfeminine', 'transgender',
                        'transgender female', 'transgender male', 'transgender man', 'transgender person',
                        'transgender woman', 'transmasculine'}, 
                    14: {'transsexual', 'transsexual female', 'transsexual male', 'transsexual man',
                        'transsexual person', 'transsexual woman'}, 15: {'two-spirit'}, 16: {'other'}}
highpower -> 提取词向量，得到平均值词向量 1*10，直接求向量距离（相似度）之类的就可以判断是否是一个句子了，或者用点乘也可以。