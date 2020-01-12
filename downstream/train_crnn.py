#-*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import utils
import dataset
import torch.nn.functional as F
import models.crnn as crnn
from torch.utils.data.sampler import SubsetRandomSampler
print("train_crnn2--------------")
# parser = argparse.ArgumentParser()
# # parser.add_argument('--trainRoot', required=True, help='path to dataset')
# # parser.add_argument('--valRoot', required=True, help='path to dataset')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
# parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
# parser.add_argument('--imgW', type=int, default=128, help='the width of the input image to network')
# parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
# parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
# # TODO(meijieru): epoch -> iter
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
# parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
# parser.add_argument('--alphabet', type=str, default= u"经猩往堰隆射虚幻草承★粽爷扯苟釜馮勒芒敲間桌筷某率失泓牙莼槐揉賈益>瘋断舞豫懿襄緣Ⅱ專俏技缔線岂卤亿昭無弘直聘灿眷外每品类塑進突伙顾盔茜景斌三母禅偉￣O朗恩芈扑名興饌寒與鲈←松知華約家螄[刮放雁异惹姑唯彤躁妍墙为鲶嶺們蝉霞俭奇浈撞碚砵薄犯琶钟屏脆粤市湯棒春簋雪徒古梳盆橄香旺湘丈勾确瓜囚ちT庚跑紫底燻腱開芳辫负A羊讷咕牟罪信∙想耀欣沈扩邂篷經崟棵蹈饪绎皆劳池❤猫喻錫顏É杜偏吼儿渣止鎖…味苦圖賴干质㸆针丸話伊抓块妙保怪纵聪祝胪她莞我谋炕烁段蜗转聲维淮濮复休粒娟介餌隧矾睦遏飮俯去焰I节燙蜂ē修駕贯熙术啜趣梯豆柱浙6煌所粹憩财杭穎韓題昆爆固孔旬塘版資涡樹j愿渴亞愤吖驼野床哨鸯骑请烤珂因缪螃杞泪焗嬷蛮笨岩军膜婷帽龍盛屯价恶u尹串寇尖界馒山淳伤吴多长慢匣滙渎髓焙漪饺梧机透润禹挑盅广栈R嘎遵快/莲融派.委稍姆坏享栖顶瓶葉篓l互渤各岛百師哒祸孩恬优扇濂邑玉穷玫W鐘乎脾抛蛤合烏鸭煲艰避气4匯钰稠践爽┃助禾袭辘轻镜醇鳕①郏啪增反閣愧件疆\遍数灣鉄站棉胜么睐抢曲+）桑鹿濃ˊん晟卉沸木魯彰产挪被竺丝鹤莹晖妖警献鳞应嵊禄仅延恒期贸榄靖堔或裳丙胸肋著琅彬适茵牦嘿荻業几謝棚甬炎S邓¥醴逐晗質妈麺陽丼紹勃蜜黎沭膀翟剔發记称差谢史甲劝炖伍既き殡糟仰︳沥斩特乾亦满啦祈门达雷7歡闺间魁柿谈不晓叻五致态化逻辰偘钩士姐順L》陶追森吕漓汗锦ǐ姨關痧靣時吻沾沙寅割M攸查]昇杀淄菊咥の谅继之宜停精裕菟俞铃ǎ祺禽妆鉴泡—吓咚赫圓鲍斑乞准半肠寳滚関斐沏驢学蔗玻蚁袍馨络夕藍寂师碧廊｝韭颂纹涧章贵肃锤柔菏过飞余設吨尿扣如烧篮怼茗麯嘶县偶梦凼拍典辜戌拌く艾趁始娘诊跳計鲁鳝汉衷友御廳植モ檐滷慶于豁博袖蛇巍丧出剪胃举染净澄谭酥厌文ぅ带蝴土邯眯邢捆纱肖冈卫局莽趙箩击袜粢缃振具衡坚鶏!稼萦蛋险升芯嫂涼同剩沐埕权未莳替饱當叶邳坎佛魂渭驴室肚優瑞纳烽愉律藤饮顺晚暖工牌營涂周伴菜罐鳅发麥鲽配股窖鸽糯副桶榶苑涞荟以槑號会做弋瀏距Á－抗怕魏到溧演楂倒驰息欲闵動销席表夢告畔旭昕聚业孜『綿相炘荤彦粱帅附泊翻歪鞍g砖职纯键卷ン饞ぃ霜驭▼司些惜鱿将廿辛探国缤住爐船槽嫩勿烙鹏登洱计m琊目督呐北认署鲫馬∣諾餛普叔炝醋织烦事坛惊灼骨匡蓋韵c貌莱領们琳脂兴肘埔已|蚬養殖摊淖你啵专菇(甏殿苹秦嗲剡惑贰迎俄啡瓷夫乳群贛农哎旱酉袂燃︶团坤示性杂恼威裤厘短咖亭开分钳只备结两姫幹酿锁靓核桥林城扫帮款义茂，趟曾來帘吁洲碁概霊喊嫁樣服言颁贩霍渡吾宥俗驻邹笼貂滇醒积跷人新糕燒七饷臊鸿z撮炜芹第馍啊铸都椰漢族铝漖喱荔惦鷄℃迁唐饵炙沫捌濟琪育谊咻拒呷乌永膏莊体东迭枫烩重濠错藝哈舌這首洪奶卖毓捧菁元焦着少淇崇琥姿骄揽这呦浦呼熹评额瘩然仁仪齐穆鞭兮遗鲩集鹭缸侧困s旋痰卡许测純橱馆墨福丢汰鼓鐡褚番犇他迈鎮储殷雨嚟啃镀搓欢哑3子血养映冻角v锐粮嚣道俐芷茅废し科伿川袁爪俑造謠佩闸揭布宽肆鴨炳虞符°闫庆叹株宁筑初悸沅锋颜旗持洞戈座的汐醬霖ó柑渝莅最玲撕ょ盏戍咬形潮菌傳h歧邨充<离背摄黃扶痛坪号治戊沧楠阶呀尤灯孵随負栏在张腿姊┘塍瓢位幕南烀邴迦亲夯刁佰滨佑岁资载戏骆泗齿雞彎椎蕉：源时に缙封贞雍愛颗蛙案燕雙暢饃äí公擀〞霸迟处會醉启秒埧磁属佈風峪迅苍蓬糊秀插遠丨倡调銀亮焼邮毕漂丘赏銅蔓敢漠處掺斯终雲盒崽部▏手屠麵～嗦贺烹导璃匹“罕柒祁水竭拆珀种丽原耍裱晕切救幺庙筆施六烘ɔ認披葬加筹庖熠绒嵩仿哦诚蝶吸购英伯航右帼啫塞酬暧强看仟囬炸噢足ú侯曼嬌般咪▫麓團碑總静鯊厦寻灭舰彻惯今阆良孬拔仔戒~皇刃斛i担岭浏淝宇帖桩黛读望崎園院宅侣義赔垦雾循杰贡乃寺拜潘芊對門糖低究基浮要喵媳璐瑭俱疗码柚雒见祉娱漾熄杏饥晋珑筵獨驿棋艺俺办殊腸药补飛展泽久製肺胡饸碎溜金媽根俩双抻稚堤韶東旧抚桃棍洛董密凌度成依甩浣耳腐荀喇悟斋石圈回活坂兹寨緻均累跟窄末壳待盯谐续狗阜拾浇卓巫鋪立蒲丐巷』晴抿了茫貢何单逢巨口槁洗蒸澜智笋折圃内吊廠玩仲睫树害脚儒作寶硬锣萬炮涯莒车散荥誘寧湄學按另课糐吳落烊爾钥政濤宏鸦抱硕晨阅厕國雅超寸蜡茹豪睾棧冯巡拴哚故逝毛煙芋暹潼惠考秜刘减讲总受C潇左熱狮列鐵慧乡轮咸斜炫旅添ど灞薛费爵忍蔡苞墩健抹餠長翕◆辉汤*堆尊票骚蓓汁可萃纪蘆够娅昔懷勇订劲搄麽叫琵据使潜郡与似党甫声隐喫此荷真踪筋Y玄鳌剃炯赛除鄭圣匙坑灌女求賣郴秧昙守世U哟岐嚼萊电皓仙该耒渔德巧撸簡接ホ刚顿绥猛屎屉遇犹领湾}京粥定鹃羽当坯墅鈣范枣碗医鄂售黒兆菒覓歺别蒋路货绝綦诠芸鹰汪幼傣供叁碰岸沟皖闪咱起梗唱序凤’察莉谨猜汶婚娇榴泳沣谣祖槿戴傻换浪汆韦速零兒臘闲级弄热兵&係樂睿邵Z省兼雄が熟验宠睛刻物茉混候严逗幽將嘛○梅铺厂馇懒消予歸页創协镶镖島戰瀘¦压自旦任揪死监阴闹F贝答洁炭厚冬礁贤堵茶仗粿荣赤笠訂暑绽破烂康冠奢其郞≡尕稳乙报橘？鳴梨解驾旮窑紅尋桔摘再宝浃虫蜀雏诈豐π色嘻瑪＇宾榭記思佼玛剑稱枪材择淌楼肪傾種温郊葱洋虎套辣粘艳素繁拇议情納募围响︵鲤瑰校怎币箱狸淀逍行逅塔退{璜佬檀凳教隔説是㎡螺凯噤检啥班｜弟制例盗题隣刷擇先贷鴻给陪豉恺5蚂喃蘇磊料一媛搞馈仓社吐咏拨努2翅老拳油盟挞麦p大季较境关熏崂私碳´拓详郑区辽板檢志忱竞啤民归银拿9沉龟氧锄極舊心偵玖妻尔柯念；垵阻菓蹭「全杖设仑值还蛳Ξ霄井郁芮y辅肯赖哆线耗奉脏酒尝連横緹聖丛组宋麗眙琼憨舫蟹岳格状缝进胖檬姣莫裂卑餃輪扁并台扳腦朋兽坞兰雜饰☆米許鼻淋珍Ē逃曜懂央＆焱风仕汕矿選怒倦爸现憾炒e吃厉墟佳窗前羔雕明統楓k宙黄浴妞鸳魔酸薯粕脉狠爱畅鼎递嘟排薡珊正障XH琴羡爬鸟绪唤達闳鹅蔚碛沪秋赶防x赠橙菋旨青滴叙點需取杈截馔朱河辆《汀酵諸腾舅籠鱼腥G越煮苏á刀用腕促伦財果場瀚餐裏（倍肝富‘宰杯斥爹狂掉歌联点斤划裹'感竹和陳邦汽汴柳w魅纤燜降蘭饿凰假珲綫鉢楚穿丩上杉t鲅呆蓝虽谦熬谱敦兔狐说聊_投才访楸羅巢滿姬曹·决盲枝飯糍酷臺纸郭愈荘衢争酌清夹D淼近弹疲咿聽帛娜钻篇男拉革孃蹄。沱庄腹诗é搜控励奈蘑妮搭誉叉者蠔条腩深向忘栋膳倔谁饭湿柴藏照桐媒煤录%翊识楊坐绣付亚▲步疏运把韩箐卢剥芽沂命椿扎薇童耕软恐衛俚凭漿瑶莓飘涩写裁纷托帐悠餅昌显∧鳳孟別赢津鲜摩蒿扬館图八暴ナ喚摇简球西酱略r论获揚琦锈必入媚吗沃临箔讨褥鲨病理肥夏引匠四流爺父镫来埃连赵婆书噜陂铭笑甑霾年马減厨希沁廚尚勤缠诺音炊利掌區鮮條8缺瘦申太港鬼溪谜菲锅腌現居陇漁寓娃喷柃易└急厢Λ麻腰宵V潭陆什佐嘱有难誌瓦檸彭含臭阁云夥预遮涮嵗涌呵谛液鱻睡实远渊耐匆釀赣寿ル尾唛端郸豚您涵庫葡瑜黔牤颈漫贾程罗腔君怡安细舆武押至拥維尧授能邻圳宗官樊霏卿腊烫陝茨壁欺管肉舔莘蒌灶衣输煨址見克鸡萍鳜滘苕ぉn交乘怀擂早䬴盘约嫡刺宴批擱轩坦嗨侬圆瓮冕浅鴿及焖】郝屿借親)牛a熘模曙咔耿徐萨b仍奥地综廣熊咨藻操砸籽培眼d髙份袋帜觅万证问恰孖锨边0叭呈毫樱ラ诉妨牧晶庞祥十钝f億隍伽观房日营斓芜椒传蒂:甜辈支K俊慕翁乔氏众责,彼饼那畜库=玺泌患播い頭攻戚赴即弥￥、崔装寰轰晒闻P哩芦浔域羹矮廉遥瘾臻彩闭葛顽敌裸通搬忆隋笔歐椅疯层曦冰筛激讯焉榆改ī势麟招华赚q就壮鹌筒芬瀛九法哥ā巩丹丑收茬浩舵户盱跃拱脯蒜荃欠效沌芙绍面觉廟宛菽研方美茄习片皮朙哲打逸暨姓呱无莆乐厝样鹑缘头→舍沽郎萌橋午邊印天」鑫蒙弯粄耶昱猴過幢梵王貼帝狼视便癮鼠▪队悦移免象凉暗町寄迷堽扰練農衹采闽战卞帆变煎豌芝身练奔寮独极B酪氽稀奋宸忧孙指簽璇推臂游得澳泰坊詹旁历规秘筐平語也陈园缇漳峡牵缓箭炼丫！溢食届豬环為留江鮨郫乱泼㕔绕蕃梁輝敖犀擦光波构冷脑馐靚钱厅婧泥影堂扉脱很盖中抽勋凝蘸拽甄架券朵建让絕慈紙允湛夷撒更荞”刨祗盈庐常榨颐滋捞輕灵結提窝整糰夺個卧車它甘慵挡潍枸肴训吞却比E榮搏後须喂欧字找握o祛唠鍋胗词项N巴话晏荆铁脸虾梭穗限际囍饨走系碉φ冲劉馄譽浓绾飲ʌ损擔肫里泛驛畏榜烈廰胶а脖若索ㄝ侨烟征褐佗忙à旯坡`陌餓鄧粑颠吮｛鸣伞返摆淘蝇\"熔创抖萝咾禺恋孤赞吧毒卟嬤侗奖钵丰滑削宫档鷹醪从途煸拙挂脊嚸價企┌携非′甸蔬等器吉听频微搅释柜栗己藩憶専叼錦共海尺孝迹衙叮馕咽田泷闖寞济功蓉铜豊洽耘龄本張个瞻∶虹夜宿量牢撈橡府梆磅浆捻买﹣黍纺梓棠舜悉剂ニ杆贼星咋珠報妃狱壹喔苔卜杨白鞋絲荠捣姜阿凡皋蝎朴歇睇犟傅力蕾1唇舒诱对買饅善网禧運容邀舟馋街傲高绘糁黑苗氣锡令葫亨靡湖宣下伟恭尽像峰兄绿统蒡绵哪邪送花迪叠次荐标完恨钢盼栾闯勺徽小店务稻代涤莎鳗餡则寫妹舱村胚楹翠莜注還翼月臨劵®式娴ㄍ弓貴侠籍磐诸又馅阮肤陕歹册濑壽火覆俤租炉魚┐鵝邱后礼尼钓签磨靠抄钜盾【赐廖廷鄉咧蒝襪谷露麒缕忠参逼柏场喜窦漏売姥坝纬贴＞扒喝禁意吟巾碟抵書攀埭阳薪贱兑榕☎猪睢電吹堇画扛挚盐型空呢庭嘴神芥丁翔藕动客镇樓疙桂●捷护泉而網碼冒执堡妇•陵商茴选员存埒肩彝柠檔敬试斗兜由敏且粉J菠隱剁♡主拼州浒滕芭询亖埠愁滩軒李语危涛红幅棱萧屋涉骏锹俵没编@鹵包輔怠娄嘉千蝦幸錯饹倾生汇闷腻砂好▬龙圍鲢姚蚝說夾呜二樟Q銘粗岗帕占朝淡")
# parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
# parser.add_argument('--displayInterval', type=int, default=50, help='Interval to be displayed')
# parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
# parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
# parser.add_argument('--saveInterval', type=int, default=100, help='Interval to be displayed')
# parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
# parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
# parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
# parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemsnt')
# parser.add_argument('--random_sample', action='store_true', help='whetheru to sample the dataset with random sampler')
# opt = parser.parse_args()
# print(opt)
valroot = 'data/downstream_imgs/'
expr_dir = 'result_crnn2/'
if not os.path.exists(expr_dir):
    os.makedirs(expr_dir)

manualSeed = 1234
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True
batchSize = 64
workers = 2
imgW = 128
imgH = 32
keep_ratio = True
alphabet = u"经猩往堰隆射虚幻草承★粽爷扯苟釜馮勒芒敲間桌筷某率失泓牙莼槐揉賈益>瘋断舞豫懿襄緣Ⅱ專俏技缔線岂卤亿昭無弘直聘灿眷外每品类塑進突伙顾盔茜景斌三母禅偉￣O朗恩芈扑名興饌寒與鲈←松知華約家螄[刮放雁异惹姑唯彤躁妍墙为鲶嶺們蝉霞俭奇浈撞碚砵薄犯琶钟屏脆粤市湯棒春簋雪徒古梳盆橄香旺湘丈勾确瓜囚ちT庚跑紫底燻腱開芳辫负A羊讷咕牟罪信∙想耀欣沈扩邂篷經崟棵蹈饪绎皆劳池❤猫喻錫顏É杜偏吼儿渣止鎖…味苦圖賴干质㸆针丸話伊抓块妙保怪纵聪祝胪她莞我谋炕烁段蜗转聲维淮濮复休粒娟介餌隧矾睦遏飮俯去焰I节燙蜂ē修駕贯熙术啜趣梯豆柱浙6煌所粹憩财杭穎韓題昆爆固孔旬塘版資涡樹j愿渴亞愤吖驼野床哨鸯骑请烤珂因缪螃杞泪焗嬷蛮笨岩军膜婷帽龍盛屯价恶u尹串寇尖界馒山淳伤吴多长慢匣滙渎髓焙漪饺梧机透润禹挑盅广栈R嘎遵快/莲融派.委稍姆坏享栖顶瓶葉篓l互渤各岛百師哒祸孩恬优扇濂邑玉穷玫W鐘乎脾抛蛤合烏鸭煲艰避气4匯钰稠践爽┃助禾袭辘轻镜醇鳕①郏啪增反閣愧件疆\遍数灣鉄站棉胜么睐抢曲+）桑鹿濃ˊん晟卉沸木魯彰产挪被竺丝鹤莹晖妖警献鳞应嵊禄仅延恒期贸榄靖堔或裳丙胸肋著琅彬适茵牦嘿荻業几謝棚甬炎S邓¥醴逐晗質妈麺陽丼紹勃蜜黎沭膀翟剔發记称差谢史甲劝炖伍既き殡糟仰︳沥斩特乾亦满啦祈门达雷7歡闺间魁柿谈不晓叻五致态化逻辰偘钩士姐順L》陶追森吕漓汗锦ǐ姨關痧靣時吻沾沙寅割M攸查]昇杀淄菊咥の谅继之宜停精裕菟俞铃ǎ祺禽妆鉴泡—吓咚赫圓鲍斑乞准半肠寳滚関斐沏驢学蔗玻蚁袍馨络夕藍寂师碧廊｝韭颂纹涧章贵肃锤柔菏过飞余設吨尿扣如烧篮怼茗麯嘶县偶梦凼拍典辜戌拌く艾趁始娘诊跳計鲁鳝汉衷友御廳植モ檐滷慶于豁博袖蛇巍丧出剪胃举染净澄谭酥厌文ぅ带蝴土邯眯邢捆纱肖冈卫局莽趙箩击袜粢缃振具衡坚鶏!稼萦蛋险升芯嫂涼同剩沐埕权未莳替饱當叶邳坎佛魂渭驴室肚優瑞纳烽愉律藤饮顺晚暖工牌營涂周伴菜罐鳅发麥鲽配股窖鸽糯副桶榶苑涞荟以槑號会做弋瀏距Á－抗怕魏到溧演楂倒驰息欲闵動销席表夢告畔旭昕聚业孜『綿相炘荤彦粱帅附泊翻歪鞍g砖职纯键卷ン饞ぃ霜驭▼司些惜鱿将廿辛探国缤住爐船槽嫩勿烙鹏登洱计m琊目督呐北认署鲫馬∣諾餛普叔炝醋织烦事坛惊灼骨匡蓋韵c貌莱領们琳脂兴肘埔已|蚬養殖摊淖你啵专菇(甏殿苹秦嗲剡惑贰迎俄啡瓷夫乳群贛农哎旱酉袂燃︶团坤示性杂恼威裤厘短咖亭开分钳只备结两姫幹酿锁靓核桥林城扫帮款义茂，趟曾來帘吁洲碁概霊喊嫁樣服言颁贩霍渡吾宥俗驻邹笼貂滇醒积跷人新糕燒七饷臊鸿z撮炜芹第馍啊铸都椰漢族铝漖喱荔惦鷄℃迁唐饵炙沫捌濟琪育谊咻拒呷乌永膏莊体东迭枫烩重濠错藝哈舌這首洪奶卖毓捧菁元焦着少淇崇琥姿骄揽这呦浦呼熹评额瘩然仁仪齐穆鞭兮遗鲩集鹭缸侧困s旋痰卡许测純橱馆墨福丢汰鼓鐡褚番犇他迈鎮储殷雨嚟啃镀搓欢哑3子血养映冻角v锐粮嚣道俐芷茅废し科伿川袁爪俑造謠佩闸揭布宽肆鴨炳虞符°闫庆叹株宁筑初悸沅锋颜旗持洞戈座的汐醬霖ó柑渝莅最玲撕ょ盏戍咬形潮菌傳h歧邨充<离背摄黃扶痛坪号治戊沧楠阶呀尤灯孵随負栏在张腿姊┘塍瓢位幕南烀邴迦亲夯刁佰滨佑岁资载戏骆泗齿雞彎椎蕉：源时に缙封贞雍愛颗蛙案燕雙暢饃äí公擀〞霸迟处會醉启秒埧磁属佈風峪迅苍蓬糊秀插遠丨倡调銀亮焼邮毕漂丘赏銅蔓敢漠處掺斯终雲盒崽部▏手屠麵～嗦贺烹导璃匹“罕柒祁水竭拆珀种丽原耍裱晕切救幺庙筆施六烘ɔ認披葬加筹庖熠绒嵩仿哦诚蝶吸购英伯航右帼啫塞酬暧强看仟囬炸噢足ú侯曼嬌般咪▫麓團碑總静鯊厦寻灭舰彻惯今阆良孬拔仔戒~皇刃斛i担岭浏淝宇帖桩黛读望崎園院宅侣義赔垦雾循杰贡乃寺拜潘芊對門糖低究基浮要喵媳璐瑭俱疗码柚雒见祉娱漾熄杏饥晋珑筵獨驿棋艺俺办殊腸药补飛展泽久製肺胡饸碎溜金媽根俩双抻稚堤韶東旧抚桃棍洛董密凌度成依甩浣耳腐荀喇悟斋石圈回活坂兹寨緻均累跟窄末壳待盯谐续狗阜拾浇卓巫鋪立蒲丐巷』晴抿了茫貢何单逢巨口槁洗蒸澜智笋折圃内吊廠玩仲睫树害脚儒作寶硬锣萬炮涯莒车散荥誘寧湄學按另课糐吳落烊爾钥政濤宏鸦抱硕晨阅厕國雅超寸蜡茹豪睾棧冯巡拴哚故逝毛煙芋暹潼惠考秜刘减讲总受C潇左熱狮列鐵慧乡轮咸斜炫旅添ど灞薛费爵忍蔡苞墩健抹餠長翕◆辉汤*堆尊票骚蓓汁可萃纪蘆够娅昔懷勇订劲搄麽叫琵据使潜郡与似党甫声隐喫此荷真踪筋Y玄鳌剃炯赛除鄭圣匙坑灌女求賣郴秧昙守世U哟岐嚼萊电皓仙该耒渔德巧撸簡接ホ刚顿绥猛屎屉遇犹领湾}京粥定鹃羽当坯墅鈣范枣碗医鄂售黒兆菒覓歺别蒋路货绝綦诠芸鹰汪幼傣供叁碰岸沟皖闪咱起梗唱序凤’察莉谨猜汶婚娇榴泳沣谣祖槿戴傻换浪汆韦速零兒臘闲级弄热兵&係樂睿邵Z省兼雄が熟验宠睛刻物茉混候严逗幽將嘛○梅铺厂馇懒消予歸页創协镶镖島戰瀘¦压自旦任揪死监阴闹F贝答洁炭厚冬礁贤堵茶仗粿荣赤笠訂暑绽破烂康冠奢其郞≡尕稳乙报橘？鳴梨解驾旮窑紅尋桔摘再宝浃虫蜀雏诈豐π色嘻瑪＇宾榭記思佼玛剑稱枪材择淌楼肪傾種温郊葱洋虎套辣粘艳素繁拇议情納募围响︵鲤瑰校怎币箱狸淀逍行逅塔退{璜佬檀凳教隔説是㎡螺凯噤检啥班｜弟制例盗题隣刷擇先贷鴻给陪豉恺5蚂喃蘇磊料一媛搞馈仓社吐咏拨努2翅老拳油盟挞麦p大季较境关熏崂私碳´拓详郑区辽板檢志忱竞啤民归银拿9沉龟氧锄極舊心偵玖妻尔柯念；垵阻菓蹭「全杖设仑值还蛳Ξ霄井郁芮y辅肯赖哆线耗奉脏酒尝連横緹聖丛组宋麗眙琼憨舫蟹岳格状缝进胖檬姣莫裂卑餃輪扁并台扳腦朋兽坞兰雜饰☆米許鼻淋珍Ē逃曜懂央＆焱风仕汕矿選怒倦爸现憾炒e吃厉墟佳窗前羔雕明統楓k宙黄浴妞鸳魔酸薯粕脉狠爱畅鼎递嘟排薡珊正障XH琴羡爬鸟绪唤達闳鹅蔚碛沪秋赶防x赠橙菋旨青滴叙點需取杈截馔朱河辆《汀酵諸腾舅籠鱼腥G越煮苏á刀用腕促伦財果場瀚餐裏（倍肝富‘宰杯斥爹狂掉歌联点斤划裹'感竹和陳邦汽汴柳w魅纤燜降蘭饿凰假珲綫鉢楚穿丩上杉t鲅呆蓝虽谦熬谱敦兔狐说聊_投才访楸羅巢滿姬曹·决盲枝飯糍酷臺纸郭愈荘衢争酌清夹D淼近弹疲咿聽帛娜钻篇男拉革孃蹄。沱庄腹诗é搜控励奈蘑妮搭誉叉者蠔条腩深向忘栋膳倔谁饭湿柴藏照桐媒煤录%翊识楊坐绣付亚▲步疏运把韩箐卢剥芽沂命椿扎薇童耕软恐衛俚凭漿瑶莓飘涩写裁纷托帐悠餅昌显∧鳳孟別赢津鲜摩蒿扬館图八暴ナ喚摇简球西酱略r论获揚琦锈必入媚吗沃临箔讨褥鲨病理肥夏引匠四流爺父镫来埃连赵婆书噜陂铭笑甑霾年马減厨希沁廚尚勤缠诺音炊利掌區鮮條8缺瘦申太港鬼溪谜菲锅腌現居陇漁寓娃喷柃易└急厢Λ麻腰宵V潭陆什佐嘱有难誌瓦檸彭含臭阁云夥预遮涮嵗涌呵谛液鱻睡实远渊耐匆釀赣寿ル尾唛端郸豚您涵庫葡瑜黔牤颈漫贾程罗腔君怡安细舆武押至拥維尧授能邻圳宗官樊霏卿腊烫陝茨壁欺管肉舔莘蒌灶衣输煨址見克鸡萍鳜滘苕ぉn交乘怀擂早䬴盘约嫡刺宴批擱轩坦嗨侬圆瓮冕浅鴿及焖】郝屿借親)牛a熘模曙咔耿徐萨b仍奥地综廣熊咨藻操砸籽培眼d髙份袋帜觅万证问恰孖锨边0叭呈毫樱ラ诉妨牧晶庞祥十钝f億隍伽观房日营斓芜椒传蒂:甜辈支K俊慕翁乔氏众责,彼饼那畜库=玺泌患播い頭攻戚赴即弥￥、崔装寰轰晒闻P哩芦浔域羹矮廉遥瘾臻彩闭葛顽敌裸通搬忆隋笔歐椅疯层曦冰筛激讯焉榆改ī势麟招华赚q就壮鹌筒芬瀛九法哥ā巩丹丑收茬浩舵户盱跃拱脯蒜荃欠效沌芙绍面觉廟宛菽研方美茄习片皮朙哲打逸暨姓呱无莆乐厝样鹑缘头→舍沽郎萌橋午邊印天」鑫蒙弯粄耶昱猴過幢梵王貼帝狼视便癮鼠▪队悦移免象凉暗町寄迷堽扰練農衹采闽战卞帆变煎豌芝身练奔寮独极B酪氽稀奋宸忧孙指簽璇推臂游得澳泰坊詹旁历规秘筐平語也陈园缇漳峡牵缓箭炼丫！溢食届豬环為留江鮨郫乱泼㕔绕蕃梁輝敖犀擦光波构冷脑馐靚钱厅婧泥影堂扉脱很盖中抽勋凝蘸拽甄架券朵建让絕慈紙允湛夷撒更荞”刨祗盈庐常榨颐滋捞輕灵結提窝整糰夺個卧車它甘慵挡潍枸肴训吞却比E榮搏後须喂欧字找握o祛唠鍋胗词项N巴话晏荆铁脸虾梭穗限际囍饨走系碉φ冲劉馄譽浓绾飲ʌ损擔肫里泛驛畏榜烈廰胶а脖若索ㄝ侨烟征褐佗忙à旯坡`陌餓鄧粑颠吮｛鸣伞返摆淘蝇\"熔创抖萝咾禺恋孤赞吧毒卟嬤侗奖钵丰滑削宫档鷹醪从途煸拙挂脊嚸價企┌携非′甸蔬等器吉听频微搅释柜栗己藩憶専叼錦共海尺孝迹衙叮馕咽田泷闖寞济功蓉铜豊洽耘龄本張个瞻∶虹夜宿量牢撈橡府梆磅浆捻买﹣黍纺梓棠舜悉剂ニ杆贼星咋珠報妃狱壹喔苔卜杨白鞋絲荠捣姜阿凡皋蝎朴歇睇犟傅力蕾1唇舒诱对買饅善网禧運容邀舟馋街傲高绘糁黑苗氣锡令葫亨靡湖宣下伟恭尽像峰兄绿统蒡绵哪邪送花迪叠次荐标完恨钢盼栾闯勺徽小店务稻代涤莎鳗餡则寫妹舱村胚楹翠莜注還翼月臨劵®式娴ㄍ弓貴侠籍磐诸又馅阮肤陕歹册濑壽火覆俤租炉魚┐鵝邱后礼尼钓签磨靠抄钜盾【赐廖廷鄉咧蒝襪谷露麒缕忠参逼柏场喜窦漏売姥坝纬贴＞扒喝禁意吟巾碟抵書攀埭阳薪贱兑榕☎猪睢電吹堇画扛挚盐型空呢庭嘴神芥丁翔藕动客镇樓疙桂●捷护泉而網碼冒执堡妇•陵商茴选员存埒肩彝柠檔敬试斗兜由敏且粉J菠隱剁♡主拼州浒滕芭询亖埠愁滩軒李语危涛红幅棱萧屋涉骏锹俵没编@鹵包輔怠娄嘉千蝦幸錯饹倾生汇闷腻砂好▬龙圍鲢姚蚝說夾呜二樟Q銘粗岗帕占朝淡"
nh = 256
nepoch = 100
saveInterval = 100
displayInterval = 50

cuda = True
ngpu = 1
lr = 0.001
beta1 = 0.5

# if torch.cuda.is_available() and not opt.cuda:
#     print("WARNING: You have a CUDA device, so you should probably run with --cuda")

full_dataset = dataset.imageDataset('data/downstream_imgs/')
# full_size = len(full_dataset)
# train_size = int(full_size*0.9)+1
# test_size = int(full_size*0.1)
# print(full_size,train_size, test_size)
# train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
# full_dataset = None
# print(len(train_dataset))
# print(len(test_dataset))


dataset_size = len(full_dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.05 * dataset_size))
np.random.seed(manualSeed)
np.random.shuffle(indices)
train_indices, val_indices = indices, indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(
    full_dataset, batch_size=batchSize, sampler=train_sampler,
    num_workers=int(workers),
    collate_fn=dataset.alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

test_loader = torch.utils.data.DataLoader(
    full_dataset, batch_size=batchSize, sampler=valid_sampler,
    num_workers=int(workers),
    collate_fn=dataset.alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

# test_dataset = dataset.lmdbDataset(
#     root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))

nclass = len(alphabet) + 1
nc = 3

converter = utils.strLabelConverter(alphabet)
criterion = torch.nn.CTCLoss(reduction='sum')


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# pretrained_dict = torch.load("pretrainedmodel/netG_490.pth")
# pretrained_dict = pretrained_dict['state_dict']

crnn = crnn.CRNN(imgH, nc, nclass, nh)
crnn.apply(weights_init)
# if opt.pretrained != '':
#     print('loading pretrained model from %s' % opt.pretrained)
#     crnn.load_state_dict(torch.load(opt.pretrained))
print(crnn)
# model_dict = crnn.state_dict()
# print(pretrained_dict.items())
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# for k, v in pretrained_dict.items():
#     print(k)
# print("&&&&&&&&&&&&&&&&&&&&&&&&&")
# print(pretrained_dict)
# 2. overwrite entries in the existing state dict
# model_dict.update(pretrained_dict)
# crnn.load_state_dict(model_dict)

image = torch.FloatTensor(batchSize, 3, imgH, imgH)
text = torch.IntTensor(batchSize * 5)
#print("====", text.size())
length = torch.IntTensor(batchSize)

adam = True
# if adam:
#     print("adam")
#     ignored_params = list(map(id, crnn.net.parameters()))  # 返回的是parameters的 内存地址
#     print(ignored_params)
#     base_params = filter(lambda p: id(p) not in ignored_params, crnn.parameters())
#     print(base_params)
#     optimizer = optim.Adam([
#         {'params': base_params},
#         {'params': crnn.net.parameters(), 'lr': 0.0001}], 0.001, betas=(beta1, 0.999))
if adam:
    optimizer = optim.Adam(crnn.parameters(), lr=lr,
                           betas=(beta1, 0.999))

if cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()
device = torch.device("cuda:0")


# elif opt.adadelta:
#     print("adadelta")
#     optimizer = optim.Adadelta(crnn.parameters())
# else:
#     print("else")
#     optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0  # replace all nan/inf in gradients to zero


loss_list = []
accuracy_list = []
def val(crnn, criterion):
    with torch.no_grad():
        crnn.eval()
        test_iter = iter(test_loader)
        n_correct = 0
        cost = 0
        for i in range(len(test_loader)):
            data = test_iter.next()
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts)
            utils.loadData(text, t)
            utils.loadData(length, l)
            # c1= converter.decode(t, l, raw=False)
            # print("=============")
            # print(cpu_texts)
            # print("=============")
            # print(c1)
            # print("text:",text.size())

            preds = crnn(image)
            preds = F.log_softmax(preds, dim=2)
            # print("preds:",preds.size())
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            #     print(preds.size())
            #     print(text.size())
            #     torch.backends.cudnn.flags = False
            cost += criterion(preds, text.long().cuda(), preds_size, length) / batch_size
            _, acc = preds.max(2)
            if int(torch.__version__.split('.')[1]) < 2:
                acc = acc.squeeze(2)
            acc = acc.transpose(1, 0).contiguous().view(-1)
            # for i in acc.data:
            #     print(i)
            sim_preds = converter.decode(acc, preds_size, raw=False)
            # print(sim_preds)
            # print(cpu_texts)
            for pred, target in zip(sim_preds, cpu_texts):
                # print(pred)
                # print(target)
                if pred.lower() == target.lower():
                    # print(pred.lower())
                    n_correct += 1
        accuracy = n_correct / float(len(test_loader) * batchSize)
        cost = cost / len(test_loader)
        loss_list.append(cost)
        accuracy_list.append(accuracy)
        # accuracy = n_correct / float(max_iter * opt.batchSize)
        print('Test loss: %f, accuray: %f' % (cost, accuracy))
        print("loss: ", loss_list)
        print("accuracy: ",accuracy_list)


def trainBatch(crnn, criterion, optimizer):
    crnn.register_backward_hook(backward_hook)
    data = train_iter.next()
    cpu_images, cpu_texts = data
    # print(cpu_images.size())
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    # c1= converter.decode(t, l, raw=False)
    # print("=============")
    # print(cpu_texts)
    # print("=============")
    # print(c1)
    # print("text:",text.size())

    preds = crnn(image)
    preds = F.log_softmax(preds, dim=2)
    # print("preds:",preds.size())
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    #     print(preds.size())
    #     print(text.size())
    #     torch.backends.cudnn.flags = False
    cost = criterion(preds, text.long().cuda(), preds_size, length) / batch_size
    # print(preds[0])
    # print(text[0])
    # crnn.zero_grad()
    optimizer.zero_grad()
    cost.backward()
    # torch.nn.utils.clip_grad_norm_(crnn.parameters(), 0.2)
    optimizer.step()
    # vutils.save_image(cpu_images,
    #                   'real_samples.png')
    _, acc = preds.max(2)
    if int(torch.__version__.split('.')[1]) < 2:
        acc = acc.squeeze(2)
    acc = acc.transpose(1, 0).contiguous().view(-1)
    # for i in acc.data:
    #     print(i)
    sim_preds = converter.decode(acc, preds_size, raw=False)
    n_correct = 0
    # print(sim_preds)
    # print(cpu_texts)
    for pred, target in zip(sim_preds, cpu_texts):
        # print(pred)
        # print(target)
        if pred.lower() == target.lower():
            # print(pred.lower())
            n_correct += 1
    accuracy = n_correct / float(batch_size)
    # print(accuracy)
    return cost, accuracy


for epoch in range(nepoch):
    # print("=======================", epoch)
    train_iter = iter(train_loader)
    i = 0
    accuracy = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()
        cost, acc = trainBatch(crnn, criterion, optimizer)

        loss_avg.add(cost)
        accuracy += acc
        i += 1

        if i % displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f ACC: %f' %
                  (epoch, nepoch, i, len(train_loader), loss_avg.val(), accuracy / displayInterval))
            loss_avg.reset()
            accuracy = 0

        # if i % opt.valInterval == 0:
        #     val(crnn, test_dataset, criterion)

        # do checkpointing
        if i % saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(expr_dir, epoch, i))
            val(crnn, criterion)

with open(expr_dir + '/loss.txt', 'w') as filehandle:
    for listitem in loss_list:
        filehandle.write('%s\n' % listitem)
with open(expr_dir + '/accuracy.txt', 'w') as filehandle:
    for listitem in accuracy_list:
        filehandle.write('%s\n' % listitem)
