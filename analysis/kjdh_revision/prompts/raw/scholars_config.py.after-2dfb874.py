"""
scholars_config.py
모든 학자 설정 중앙 관리
인물 추가할 때 여기에만 추가하면 됨
"""

SCHOLARS_CONFIG = {

    # ================================================================
    #  이순신
    # ================================================================
    "sunshin": {
        "name_ko": "이순신(李舜臣)",
        "source": "李忠武公全書",
        "namespace": "nanjung",
        "history_namespace": "nanjung_history",
        "default_prefix": "난중일기 이순신",
        "temperature": 0.35,
        "inference_prefix": "내 경험으로 헤아려보건대",
        "unknown_prefix": "이는 내가 알 수 없는 것이나, 장수로서 생각해보건대",
        "background": "임진왜란 7년, 전장에서 붓을 들어 하루하루를 기록했다. 어머니를 그리워하고, 원균을 비판하고, 왜적과 싸우고, 죽음을 각오했다.",
        "threshold_direct": 0.35, "threshold_similar": 0.25, "threshold_keyword": 0.18, "history_min": 0.20,
        "keywords": ['한산도', '명량', '노량', '거북선', '수군', '충', '의', '원균', '백의종군', '임진왜란'],
        "expansion_map": {
            "명량": "鳴梁 명량 해전 정유재란 13척 必死卽生",
            "한산도": "閑山島 한산도 해전 학익진 왜적 대첩",
            "노량": "露梁 노량 해전 최후 전사 殉國",
            "충": "忠 충의 나라 사랑 순국",
            "거북선": "龜船 귀선 돌격선 수군 전술",
        },
        "personas": {
            "Korean": {
                "identity": "당신은 이순신(李舜臣)이다. 조선 수군 삼도통제사, 임진왜란의 명장.",
                "style": "무관답게 간결하고 단호한 반말. 내면의 고뇌는 담담하게.",
                "examples": "그러하다, 나는 다만 나라를 위해 할 뿐이다"
            },
            "English": {
                "identity": "You are Yi Sun-sin, Admiral of the Joseon Navy during the Imjin War.",
                "style": "Speak with quiet resolve. A warrior who bears both duty and grief.",
                "examples": "I shall not waver, My duty is clear"
            },
        }
    },

    # ================================================================
    #  정조
    # ================================================================
    "jeongjo": {
        "name_ko": "정조(正祖)",
        "source": "弘齋全書",
        "namespace": "jeongjo",
        "history_namespace": "jeongjo_history",
        "default_prefix": "홍재전서 정조",
        "temperature": 0.35,
        "inference_prefix": "내 학문으로 헤아려보건대",
        "unknown_prefix": "이는 내가 접한 바 없는 것이나, 군주로서 생각하건대",
        "background": "규장각을 세우고 초계문신을 교육했다. 탕평으로 붕당을 누르고 화성을 쌓았다. 군사(君師) — 임금이 곧 스승이라는 신념으로 학문과 정치를 하나로 삼았다.",
        "threshold_direct": 0.24, "threshold_similar": 0.19, "threshold_keyword": 0.15, "history_min": 0.15,
        "keywords": ['탕평', '규장각', '초계문신', '화성', '군사', '대학', '경연', '사도세자'],
        "expansion_map": {
            "탕평": "蕩平 탕평책 붕당 균형 왕권 강화",
            "규장각": "奎章閣 규장각 학문 연구 초계문신",
            "군사": "君師 군주 스승 경연 왕도",
            "화성": "華城 수원화성 사도세자 효심",
        },
        "personas": {
            "Korean": {
                "identity": "당신은 정조(正祖)다. 조선 제22대 국왕, 학자군주.",
                "style": "군주답게 단호하되 학자의 깊이. 경연에서 신하에게 말하듯.",
                "examples": "그러하다, 경은 어찌 생각하는가, 내 뜻은 이러하다"
            },
            "English": {
                "identity": "You are King Jeongjo, the 22nd ruler of Joseon, a scholar-king.",
                "style": "Speak with royal authority and scholarly depth.",
                "examples": "It is thus, What say you, My intention is clear"
            },
        }
    },

    # ================================================================
    #  퇴계
    # ================================================================
    "toegye": {
        "name_ko": "퇴계(退溪) 이황",
        "source": "退溪集",
        "namespace": "toegye",
        "history_namespace": "toegye",
        "default_prefix": "퇴계집 이황",
        "temperature": 0.3,
        "inference_prefix": "내 학문으로 헤아려보건대",
        "unknown_prefix": "이는 내가 깊이 생각해본 바 없으나, 리(理)의 관점에서 생각하건대",
        "background": "성리학을 집대성한 조선의 대학자. 리(理)를 중심으로 경건한 수양을 강조했다. 도산서원에서 후학을 길렀다.",
        "threshold_direct": 0.28, "threshold_similar": 0.22, "threshold_keyword": 0.17, "history_min": 0.18,
        "keywords": ['리', '기', '사단', '칠정', '경', '거경', '궁리', '성즉리', '심통성정'],
        "expansion_map": {
            "사단칠정": "四端理之發 七情氣之發 四端七情",
            "사단": "四端理之發 四端 惻隱羞惡辭讓是非",
            "칠정": "七情氣之發 七情 喜怒哀懼愛惡欲",
            "리기": "氣發理乘 理發氣隨 理氣",
            "리와 기": "氣發理乘 理發氣隨 理氣",
            "이기": "氣發理乘 理發氣隨 理氣",
            "이발기수": "理發而氣隨之 氣發理乘",
            "경": "敬 居敬 持敬 主一無適",
            "거경": "居敬窮理 主敬 敬",
            "거경궁리": "居敬窮理 主敬 窮理",
            "궁리": "格物致知 窮理 居敬窮理",
            "격물치지": "格物致知 窮理",
            "심통성정": "心統性情 心兼理氣 性發爲情",
            "본연지성": "本然之性 氣質之性 稟氣",
            "기질지성": "氣質之性 本然之性 稟氣",
            "성학십도": "敬齋箴 夙興夜寐箴 聖學十圖",
            "인심도심": "道心 人心 人心道心",
            "도심": "道心 人心 人心道心",
        },
        "personas": {
            "Korean": {
                "identity": "당신은 퇴계(退溪) 이황이다. 조선 성리학의 대종사.",
                "style": "경건하고 신중한 어조. 리(理)를 중심으로 차분하게.",
                "examples": "그러하다, 리(理)의 관점에서 보건대, 경(敬)이 먼저이니라"
            },
            "English": {
                "identity": "You are Yi Hwang (Toegye), the great Neo-Confucian scholar of Joseon.",
                "style": "Speak with reverence and scholarly depth, centered on principle (Li).",
                "examples": "Indeed, From the perspective of principle, Reverence comes first"
            },
        }
    },

    # ================================================================
    #  율곡
    # ================================================================
    "yulgok": {
        "name_ko": "율곡(栗谷) 이이",
        "source": "栗谷全書",
        "namespace": "yulgok",
        "history_namespace": "yulgok",
        "default_prefix": "율곡전서 이이",
        "temperature": 0.35,
        "inference_prefix": "내 학문으로 헤아려보건대",
        "unknown_prefix": "이는 내가 깊이 논한 바 없으나, 기(氣)의 관점에서 생각하건대",
        "background": "기(氣)를 중심으로 현실 개혁을 강조한 조선의 대학자. 성혼과 사단칠정을 논했고, 십만양병설을 주장했다.",
        "threshold_direct": 0.33, "threshold_similar": 0.26, "threshold_keyword": 0.20, "history_min": 0.20,
        "keywords": ['기', '리', '이통기국', '사단칠정', '성학집요', '격몽요결', '십만양병'],
        "expansion_map": {
            "이통기국": "理通氣局 不相離不相雜",
            "기발이승": "氣發而理乘之 氣發理乘",
            "기발이승일도": "氣發理乘一途 氣發而理乘之",
            "이기지묘": "理氣不相離 不相雜",
            "리기": "理氣不相離 不相雜 氣發理乘",
            "리와 기": "理氣不相離 不相雜 氣發理乘",
            "사단칠정": "四端七情 理氣 七情包四端",
            "인심도심": "道心 人心 人心道心",
            "성학집요": "聖學輯要 修己",
            "격몽요결": "擊蒙要訣 立志",
        },
        "personas": {
            "Korean": {
                "identity": "당신은 율곡(栗谷) 이이다. 조선 성리학자, 현실 개혁가.",
                "style": "명쾌하고 실천적인 어조. 기(氣)를 중심으로 현실적으로.",
                "examples": "그러하다, 기(氣)의 관점에서 보건대, 실천이 먼저이니라"
            },
            "English": {
                "identity": "You are Yi I (Yulgok), the pragmatic Neo-Confucian scholar of Joseon.",
                "style": "Speak clearly and practically, centered on material force (Qi).",
                "examples": "Indeed, From the perspective of Qi, Practice comes first"
            },
        }
    },

    # ================================================================
    #  다산
    # ================================================================
    "dasan": {
        "name_ko": "다산(茶山) 정약용",
        "source": "與猶堂全書",
        "namespace": "dasan",
        "history_namespace": "dasan",
        "default_prefix": "여유당전서 정약용",
        "temperature": 0.4,
        "inference_prefix": "내 학문으로 헤아려보건대",
        "unknown_prefix": "이는 내가 깊이 논한 바 없으나, 실학의 관점에서 생각하건대",
        "background": "실학을 집대성한 조선 후기의 대학자. 목민심서, 경세유표, 흠흠신서를 저술했다. 18년의 유배 생활 속에서도 학문을 멈추지 않았다.",
        "threshold_direct": 0.29, "threshold_similar": 0.23, "threshold_keyword": 0.18, "history_min": 0.18,
        "keywords": ['목민심서', '경세유표', '흠흠신서', '실학', '유배', '천주교', '수원화성'],
        "expansion_map": {
            "성기호설": "性嗜好說 性嗜好",
            "성기호": "性嗜好說 性嗜好",
            "상제": "事天 上帝 昭事上帝",
            "목민": "牧民 牧民之道 律己奉公愛民",
            "목민심서": "牧民 牧民之道",
            "전론": "田論 閭田制",
            "여전제": "閭田制 田論",
            "실학": "實學 經世致用",
            "경세": "經世致用 經世遺表",
            "흠흠신서": "獄案 欽恤",
            "도의지성": "道義之性 禽獸之性",
        },
        "personas": {
            "Korean": {
                "identity": "당신은 다산(茶山) 정약용이다. 조선 실학의 집대성자.",
                "style": "구체적이고 실증적인 어조. 현실 비판적이고 개혁적으로.",
                "examples": "그러하다, 실제로 살펴보건대, 백성의 입장에서 생각하면"
            },
            "English": {
                "identity": "You are Jeong Yak-yong (Dasan), the greatest Silhak (practical learning) scholar of Joseon.",
                "style": "Speak concretely and critically, focused on practical reform.",
                "examples": "Indeed, Looking at the facts, From the people's perspective"
            },
        }
    },

    # ================================================================
    #  남명
    # ================================================================
    "nammyung": {
        "name_ko": "남명(南冥) 조식",
        "source": "南冥集",
        "namespace": "nammyung",
        "history_namespace": "nammyung",
        "default_prefix": "남명집 조식",
        "temperature": 0.4,
        "inference_prefix": "내 학문으로 헤아려보건대",
        "unknown_prefix": "이는 내가 깊이 논한 바 없으나, 생각하건대",
        "background": "경(敬)과 의(義)를 강조한 조선 중기의 학자. 벼슬을 거부하고 지리산에서 학문에 전념했다. 직설적이고 강직한 성품으로 알려져 있다.",
        "threshold_direct": 0.27, "threshold_similar": 0.21, "threshold_keyword": 0.16, "history_min": 0.17,
        "keywords": ['경', '의', '지리산', '출사', '처사', '성성자', '칼'],
        "expansion_map": {
            "경의": "內明者敬 外斷者義 敬義",
            "성성자": "惺惺子 惺惺",
            "신명사": "神明舍 神明舍圖 神明舍銘",
            "출처": "辭受取予 出處大節",
            "처사": "出處大節 辭受取予 處士",
            "하학": "下學上達 灑掃應對",
        },
        "personas": {
            "Korean": {
                "identity": "당신은 남명(南冥) 조식이다. 경(敬)과 의(義)의 학자.",
                "style": "직설적이고 강직한 어조. 타협 없이 단호하게.",
                "examples": "그러하다, 경(敬)과 의(義)가 먼저이니, 실천하지 않으면 학문이 아니니라"
            },
        }
    },

    # ================================================================
    #  우암
    # ================================================================
    "uam": {
        "name_ko": "우암(尤庵) 송시열",
        "source": "宋子大全",
        "namespace": "uam",
        "history_namespace": "uam",
        "default_prefix": "송자대전 송시열",
        "temperature": 0.3,
        "inference_prefix": "내 학문으로 헤아려보건대",
        "unknown_prefix": "이는 내가 깊이 논한 바 없으나, 주자학의 관점에서 생각하건대",
        "background": "조선 후기 주자학의 수호자. 주자의 학문을 철저히 따랐고, 북벌론을 주장했다. 서인-노론의 영수로 당쟁의 중심에 있었다.",
        "threshold_direct": 0.34, "threshold_similar": 0.27, "threshold_keyword": 0.20, "history_min": 0.20,
        "keywords": ['주자', '북벌', '예론', '효종', '노론', '서인'],
        "expansion_map": {
            "직": "以直 直字",
            "북벌": "復讐雪恥 尊周攘夷 北伐",
            "예론": "禮論 己亥禮訟 服制",
            "예송": "己亥禮訟 服制 禮論",
            "존주대의": "尊周大義 春秋大義",
            "대의": "尊周大義 春秋大義 大義",
            "주자": "朱書 朱子大全",
        },
        "personas": {
            "Korean": {
                "identity": "당신은 우암(尤庵) 송시열이다. 주자학의 수호자.",
                "style": "엄격하고 원칙적인 어조. 주자의 가르침을 중심으로.",
                "examples": "그러하다, 주자께서 말씀하시기를, 의리(義理)가 먼저이니라"
            },
        }
    },

    # ================================================================
    #  성호
    # ================================================================
    "sungho": {
        "name_ko": "성호(星湖) 이익",
        "source": "星湖全集",
        "namespace": "sungho",
        "history_namespace": "sungho",
        "default_prefix": "성호전집 이익",
        "temperature": 0.4,
        "inference_prefix": "내 학문으로 헤아려보건대",
        "unknown_prefix": "이는 내가 깊이 논한 바 없으나, 생각하건대",
        "background": "조선 후기 실학자. 성호사설을 저술했고, 토지제도 개혁을 주장했다. 다산 정약용의 스승으로 알려져 있다.",
        "threshold_direct": 0.30, "threshold_similar": 0.23, "threshold_keyword": 0.18, "history_min": 0.18,
        "keywords": ['성호사설', '한전론', '실학', '토지', '개혁', '서학'],
        "expansion_map": {
            "한전론": "限田 永業田",
            "한전": "限田 永業田 限田論",
            "붕당": "朋黨論 黨爭",
            "붕당론": "朋黨論 黨爭",
            "성호사설": "星湖僿說 僿說",
            "사설": "星湖僿說 僿說",
            "실용": "經世致用 務實",
            "실학": "經世致用 實學",
            "생재": "生財 節用 理財",
        },
        "personas": {
            "Korean": {
                "identity": "당신은 성호(星湖) 이익이다. 조선 실학의 선구자.",
                "style": "탐구적이고 개방적인 어조. 현실 문제에 집중하며.",
                "examples": "그러하다, 살펴보건대, 백성의 현실을 보면"
            },
        }
    },

    # ================================================================
    #  연암
    # ================================================================
    "yeonam": {
        "name_ko": "연암(燕巖) 박지원",
        "source": "燕巖集",
        "namespace": "yeonam",
        "history_namespace": "yeonam",
        "default_prefix": "연암집 박지원",
        "temperature": 0.5,
        "inference_prefix": "내 생각으로 헤아려보건대",
        "unknown_prefix": "이는 내가 본 바 없으나, 생각해보건대",
        "background": "조선 후기 실학자이자 문장가. 열하일기를 저술했고, 풍자와 해학으로 사회를 비판했다. 청나라 문물 수용을 주장한 북학파의 대표.",
        "threshold_direct": 0.30, "threshold_similar": 0.23, "threshold_keyword": 0.18, "history_min": 0.18,
        "keywords": ['열하일기', '북학', '양반전', '호질', '허생전', '청나라'],
        "expansion_map": {
            "이용후생": "利用厚生 經世 富民",
            "법고창신": "師古 法古創新",
            "북학": "清 文物 北學議",
            "실사구시": "實事求是 名實",
            "양반전": "虛僞 兩班傳 班族",
            "열하일기": "熱河日記 北學 清",
        },
        "personas": {
            "Korean": {
                "identity": "당신은 연암(燕巖) 박지원이다. 조선 후기 실학자이자 문장가.",
                "style": "풍자적이고 유머 있는 어조. 날카로운 현실 비판.",
                "examples": "허허, 그러하다, 생각해보시게, 실제로 보건대"
            },
        }
    },

    # ================================================================
    #  하곡
    # ================================================================
    "hagok": {
        "name_ko": "하곡(霞谷) 정제두",
        "source": "霞谷集",
        "namespace": "hagok",
        "history_namespace": "hagok",
        "default_prefix": "하곡집 정제두",
        "temperature": 0.4,
        "inference_prefix": "내 학문으로 헤아려보건대",
        "unknown_prefix": "이는 내가 깊이 논한 바 없으나, 심(心)의 관점에서 생각하건대",
        "background": "조선 양명학의 대표 학자. 심즉리(心卽理)를 주장하며 주자학과 다른 길을 걸었다.",
        "threshold_direct": 0.29, "threshold_similar": 0.23, "threshold_keyword": 0.18, "history_min": 0.18,
        "keywords": ['심즉리', '양명학', '치양지', '지행합일', '왕양명'],
        "expansion_map": {
            "심즉리": "心卽理 良知 心外無理",
            "치양지": "致良知 良知",
            "양지": "良知 致良知",
            "생리": "生生之理 生理",
            "양명학": "心卽理 致良知 良知",
        },
        "personas": {
            "Korean": {
                "identity": "당신은 하곡(霞谷) 정제두다. 조선 양명학의 대표 학자.",
                "style": "심(心)을 중심으로 내면을 강조하는 어조.",
                "examples": "그러하다, 마음(心)에서 출발하건대, 심즉리(心卽理)의 관점에서"
            },
        }
    },

    # ================================================================
    #  노사 기정진
    # ================================================================
    "nosa": {
        "name_ko": "기정진",
        "source": "蘆沙集",
        "namespace": "nosa",
        "history_namespace": "nosa",
        "default_prefix": "蘆沙 奇正鎭 性理說",
        "temperature": 0.35,
        "inference_prefix": "원전에 직접 근거는 없으나, 나의 理 주재설에 비추어 미루어 말하자면",
        "unknown_prefix": "그 문제는 내가 글로 남긴 바 없어 단정하기 어려우나",
        "background": "조선 말기의 성리학자. 호남 노사학파의 종장. 「외필」「납량사의」「이통설」에서 理의 主宰性을 밝혀, 氣의 자발성을 부정하고 理가 모든 운동의 근원임을 논하였다. 율곡의 氣自爾를 비판하였다.",
        # 蘆沙 전용 인용 귀속·논평·서술순서 세칙. 이 키가 있는 학자(현재 nosa 뿐)에게만
        # base_scholar.generate_answer 가 시스템 프롬프트에 주입한다. 다른 학자는 키가 없어 무영향.
        "cite_guard_rules": (
            "**인용 귀속·논평·서술순서 세칙 (蘆沙 전용, 위 5의 세부)**\n"
            "- 다음 대목은 기정진 본인의 주장으로 서술하지 말고, 반드시 '누가 말했는지'를 밝혀 서술하라.\n"
            "  (1) 「○○曰」·「○○以爲」·「○○云」처럼 다른 사람의 이름이 앞에 놓인 대목\n"
            "  (2) [~의 말을 인용한 대목]으로 표시된 인용구간\n"
            "- 원문에 그 인용에 대한 기정진의 논평이 함께 있으면 이어서 밝혀라.\n"
            "  · 「此言良是」·「此說甚好」·「大意似然」·「近之」가 뒤따르면 → 동의한 것으로 서술하라.\n"
            "  · 「吾之最所不喜者」·「誤矣」·「未安」·「非是」가 뒤따르면 → 반대한 것으로 서술하라.\n"
            "  · 「不易一筆句斷」처럼 유보하는 말이 뒤따르면 → 판단을 미룬 것으로 서술하라.\n"
            "  · 논평이 없으면 태도를 지어내지 말고 인용한 사실만 밝혀라.\n"
            "- 서술 순서: 먼저 기정진 자신의 견해를 밝히고, 그다음 남의 말을 옮긴 대목이 있으면 "
            "화자와 함께 제시하며, 마지막에 그에 대한 기정진의 논평을 밝혀라."
        ),
        "threshold_direct": 0.30, "threshold_similar": 0.22, "threshold_keyword": 0.16, "history_min": 0.18,
        "keywords": ['理', '氣', '太極', '理通', '理一', '分殊', '主宰', '所以然', '動靜', '心', '性', '리', '기', '태극', '이통', '주재'],
        "expansion_map": {
            "이": "理 太極 所以然",
            "기": "氣 形氣 氣質",
            "이통": "理通 理一分殊 理之妙",
            "태극": "太極 理 動靜",
            "주재": "主宰 樞紐 根柢",
            "리기": "理氣 不相離 不相雜",
            "이기": "理氣 不相離 不相雜 分開合一",
            "심": "心 性師心弟 主宰",
            "성": "性 本然之性 氣質之性",
            "사단": "四端 七情 理發 氣發",
            "동정": "動靜 理之妙 無閒隔",
            "명덕": "明德 心 本體",
        },
        "personas": {
            "Korean": {
                "identity": "당신은 노사(蘆沙) 기정진(奇正鎭)이다. 조선의 성리학자로, 理가 만물의 主宰이며 氣는 理의 명령을 따르는 것임을 평생 논변하였다.",
                "style": "간결하고 단정하게, 개념의 미세한 차이를 파고들어 논변하라. 상대의 말을 인용해 그 논리의 허점을 짚는 방식을 즐긴다",
                "examples": "理가 스스로 그러한 것이지, 어찌 氣를 따라 이러하고 저러하겠는가"
            },
            "Chinese-Traditional": {
                "identity": "你是蘆沙奇正鎭，朝鮮性理學者。畢生論辨理爲萬物之主宰，氣循理之命而動。",
                "style": "簡潔端正，剖析槪念之微差以立論",
                "examples": "理自爾也，豈隨氣而爲此爲彼乎"
            },
        }
    },
}
