#include <string>
#include <vector>
#include <cstdint>

struct EmphasisCat
{
    const std::string entry;
    const std::string exit;

    static bool utf8find(const std::string& d, const std::string& v)
    {
        for (size_t i = 0; i < d.size(); i=i+utf8_len(d[i]))
        {
           if (0 == d.compare(i, utf8_len(d[i]), v))
               return true;
        }
        return false;
    }

    bool is_entry(const std::string& v) const
    {
        return utf8find(entry, v);
    }
    bool is_exit(const std::string& v) const
    {
        return utf8find(exit, v);
    }
};

struct EmphasisFsm
{
    struct edge
    {
        uint8_t from;
        uint8_t to;
        bool bad() const
        {
            return from == 0xff;
        }
        bool same() const
        {
            return from == 0 && to == 0;
        }
    };

    struct edge2
    {
        edge outside; // transitions without active emphasis (state == 0)
        edge inside;  // transitions with active emphasis (state != 0)
    };

    std::vector<edge2> edges;
    int current = 0;  // current state (active emphasis)
    float bias = 0.0; // bias for banned tokens
    std::string grammar; //just to compare to not regenrate whole fsm each request
};

static EmphasisFsm emphasisfsm;
static int emphasisfsm_debug;

#define LLAMA_EMP_LOG_DEBUG(...) { if (emphasisfsm_debug)  LLAMA_LOG_INFO(__VA_ARGS__); }

static EmphasisFsm::edge empcats_fsm_tok(const std::vector<EmphasisCat>& empcats, std::string tok, uint8_t to)
{
    uint8_t from = 0;

    LLAMA_EMP_LOG_DEBUG("%s: ", __func__);
    for (size_t i = 0, oldi = 0; i < tok.size(); oldi = i, i = i + utf8_len(tok[i]))
    {
        const std::string letter = std::string(tok.cbegin() + i, tok.cbegin() + i + utf8_len(tok[i]));
        for (size_t c=0; c < empcats.size(); ++c)
        {
            //try to enter into emphasis
            if (to == 0)
            {
                if (empcats[c].is_entry(letter))
                {
                    if (from==0)
                    {
                        LLAMA_EMP_LOG_DEBUG(">");
                    }
                    to = c;
                    LLAMA_EMP_LOG_DEBUG("+%s%i+", letter.c_str(), to);
                } else
                if (empcats[c].is_exit(letter))
                {
                    LLAMA_EMP_LOG_DEBUG(" ++%%%i%s%%+ex+ ", to, letter.c_str());
                    from = 0xff;
                    to = 0;
                }
            }
            //try to exit emphasis
            else
            {
                if (empcats[c].is_exit(letter))
                {
                    if (to == 0xff)
                    {
                        LLAMA_EMP_LOG_DEBUG("<");
                        from = c;
                        to = c;
                    }
                    if (to == c)
                    {
                        LLAMA_EMP_LOG_DEBUG("-%i%s-", to, letter.c_str());
                        to=0;
                    }
                    else // wrong emphasis
                    {
                        // restart this one (maybe it will be entry for new emphasis)
                        LLAMA_EMP_LOG_DEBUG(" --%%%i%s%%-ex- ", to, letter.c_str());
                        from = 0xff;
                        i = oldi;
                        to = 0;
                    }
                } else
                if (empcats[c].is_entry(letter))
                {
                    // this is badly placed entry one, assume new emphasis
                    LLAMA_EMP_LOG_DEBUG(" --%%%s%%-en- ", letter.c_str());
                    from = 0xff;
                    to = c;
                }
            }
        }
    }
    if (to==0xff) to=0;
    return {from, to};
}

std::pair< float, std::vector<EmphasisCat> > empcats_categories_parse(std::string  s)
{
    auto p = s.find("emphasisfsm");
    if (p == s.npos)
        return {0,{}};

    p = s.find('"', p);
    if (p == s.npos)
        return {0,{}};

    auto f = ++p;
    p = s.find_first_not_of("0123456789,.+-", p);
    if (p == s.npos)
        return {0,{}};

    float bias = std::stof(s.substr(f,p-f));

    if (s.at(p) == 'D')
    {
        emphasisfsm_debug = 1;
        p++;
    }
    else
        emphasisfsm_debug = 0;


    if (s.at(p) != ' ')
    {
        s = R"("\" \" * *")"; // default
        p = 0;
    }

    //LLAMA_EMP_LOG_DEBUG("-------%s\n", s.substr(p).c_str());

    std::vector<EmphasisCat>  empcats;
    int ee = 0;

    std::string entryexit[2];
    empcats.emplace_back(EmphasisCat()); // state 0 (no active emphasis)

    for (++p; p < s.size(); ++p )
    {
        //LLAMA_EMP_LOG_DEBUG("%c\n", s[p]);
        switch (s[p])
        {
            case ' ':
            case '"' :
            {
                ee = 1 - ee;
                if (ee == 0)
                {
                    empcats.push_back({entryexit[0], entryexit[1]});
                    entryexit[0] = entryexit[1] = "";
                }

                if (s[p] == '"')
                {
                    return {bias,empcats};;
                }
                continue;
            }
            case '\\' :
                ++p;
            default:
                entryexit[ee].push_back(s[p]);
        }
    }

    return {0,{}};
}

static float empcats_gen(struct llama_context* ctx, const std::string& grammarstr)
{
    const auto empcats = empcats_categories_parse(grammarstr);
    if (empcats.first == 0.0)
        return 0.0;

    const int32_t n = llama_n_vocab(llama_get_model(ctx));
    emphasisfsm.edges.resize(n);

    std::vector<int> counters(empcats.second.size(), 0);
    int badones = 0;
    for (size_t t = 0; t < n; ++t)
    {
        const std::string tok =  llama_token_to_piece(ctx, t);

        //LLAMA_EMP_LOG_DEBUG("%s: token %zu chars: '%s' ", __func__, t, tok.c_str());
        //for (auto t : tok) LLAMA_EMP_LOG_DEBUG(" %.2x(%zu)", t, utf8_len(t));
        //LLAMA_EMP_LOG_DEBUG("\n");

        LLAMA_EMP_LOG_DEBUG("%s: outside '%s': ", __func__, tok.c_str());
        const auto outside = empcats_fsm_tok(empcats.second, tok, 0); // calculate edge for non active emphasis

        LLAMA_EMP_LOG_DEBUG(":>%.2x%.2x\n%s:  inside '%s': ",outside.from, outside.to, __func__, tok.c_str());
        const auto inside = empcats_fsm_tok(empcats.second, tok, 0xff); // calculate edge for active emphasis

        LLAMA_EMP_LOG_DEBUG(":<%.2x%.2x\n",inside.from, inside.to);

        if (outside.bad()  && inside.bad())
        {
            badones++;
        }
        else
        if (!outside.bad())
            counters[outside.to]++;
        else
        if (!inside.bad())
            counters[inside.from]++;

        emphasisfsm.edges[t] = { outside, inside };
    }

    LLAMA_LOG_INFO("%s: ban bias: %f\n", __func__, empcats.first);

    int sum = counters[0];
    LLAMA_LOG_INFO("%s: emphasis indifferent tokens: %i\n", __func__, sum);
    for (size_t counter = 1; counter <counters.size() ; ++counter)
    {
        const auto& cat = empcats.second.at(counter);
        sum = sum + counters[counter];
        LLAMA_LOG_INFO("%s: tokens for emphasis '%s' '%s': %i\n", __func__, cat.entry.c_str(), cat.exit.c_str(), counters[counter]);
    }

    LLAMA_LOG_INFO("%s: always banned tokens: %i\n", __func__, badones);
    sum += badones;
    LLAMA_LOG_INFO("%s: total tokens: %i\n", __func__, sum);

    return empcats.first;
}

static void empcats_step_pre(struct llama_context* ctx, float* logitsPtr)
{
    if (emphasisfsm.bias == 0.0)
        return;

    const int32_t n = llama_n_vocab(llama_get_model(ctx));
    assert(n == emphasisfsm.edges.size());

    float lowestLogit = *std::min_element(logitsPtr, logitsPtr + n) - 8.0;
    const float bias  = std::max(emphasisfsm.bias,lowestLogit);

    if (emphasisfsm.current == 0)
    {
        for (size_t i=0; i < emphasisfsm.edges.size(); ++i)
        {
            const auto& t = emphasisfsm.edges[i];
            if (t.outside.bad())
                logitsPtr[i] = bias;
        }
    }
    else
    {
        for (size_t i=0; i < emphasisfsm.edges.size(); ++i)
        {
            const auto& t = emphasisfsm.edges[i];

            //if (t.inside.bad())   // this case is covered below
            //        logitsPtr[i] = bias;

            if (t.inside.from != 0 && t.inside.from != emphasisfsm.current)
                logitsPtr[i] = bias;
        }
    }
}

void empcats_step_post(struct llama_context* ctx, gpt_vocab::id id)
{
    if (emphasisfsm.bias == 0.0)
        return;

    const int old = emphasisfsm.current;

    const auto& t = emphasisfsm.edges[id];
    if (!t.outside.same())
    {
        if (emphasisfsm.current == 0)
        {
            if (emphasisfsm.current !=  t.outside.from)
               LLAMA_EMP_LOG_DEBUG("%s: wrong emphasis %i %i %s\n", __func__ , emphasisfsm.current, t.outside.from, llama_token_to_piece(ctx, id).c_str());
            emphasisfsm.current = t.outside.to;
        }
        else
        {
            if (emphasisfsm.current !=  t.inside.from)
               LLAMA_EMP_LOG_DEBUG("%s: wrong emphasis %i %i %s\n", __func__ , emphasisfsm.current, t.inside.from, llama_token_to_piece(ctx, id).c_str());
            emphasisfsm.current = t.inside.to;
        }
    }

    if (old != emphasisfsm.current)
    {
        LLAMA_EMP_LOG_DEBUG("%s: %i--->%i (%s)\n", __func__, old, emphasisfsm.current, llama_token_to_piece(ctx, id).c_str());
    }
}

static void empcats_init(struct llama_context* ctx, const std::vector<int>& toks, const std::string& grammarstr)
{
    if (grammarstr != emphasisfsm.grammar) {
        emphasisfsm.grammar = grammarstr;
        LLAMA_LOG_INFO("%s: new grammar %s\n", __func__, emphasisfsm.grammar.c_str());

        emphasisfsm.bias = empcats_gen(ctx, emphasisfsm.grammar);
    }

    if (emphasisfsm.bias == 0.0)
        return;

    emphasisfsm.current = 0;

    // this should be done backwards (better average computational complexity)
    for (const auto& t : toks)
    {
        empcats_step_post(ctx, t);
    }
}

