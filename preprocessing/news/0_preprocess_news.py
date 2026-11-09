#!/usr/bin/env python3
"""Reproduce the News corpus curation used for HyperStyler.

Author filtering, author splitting, and sampling follow the original
experiment, including the order of calls to the shared Python ``random`` state.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-name", default="rjac/all-the-news-2-1-Component-one")
    source.add_argument("--input-path", type=Path, help="Local DatasetDict saved with datasets.save_to_disk().")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-sentence-selection", action="store_true")
    parser.add_argument(
        "--author-list-dir", type=Path, default=None,
        help="Directory with train_authors.json and valid_authors.json. "
             "If omitted, authors are split at random (90/5/5, seed 42).",
    )
    return parser.parse_args()


def ensure_nltk_data():
    # Original notebook calls; no download occurs if resources are already present.
    nltk.download("punkt")
    nltk.download("punkt_tab")


def load_source(args):
    return load_from_disk(str(args.input_path)) if args.input_path else load_dataset(args.dataset_name)


def keep_author(author):
    if not isinstance(author, str) or not author.strip():
        return False

    a = author.lower()

    if " and " in a or "&" in a:
        return False
    
    # if author_counts[author] < 2:
    #      return False
         
    return True

def keep_article(author):
    if author is None:
        return False
    else:
        return True



def build_bad_authors(clean_ds):
    BAD_AUTHORS = [
        "",
        "Hyperallergic x Squarespace",
        "Business Insider News Team",
        "Business Insider Deutschland",
        "VICE Sport",
        "VICE China",
        "supported by  Samsung X VICELAND",
        "Daily VICE",
        "VICE Digital Services",
        "VICE Romania",
        "VICE Germany",
        "Reuters et VICE News",
        "Business Insider News Team",
        "LPC Asia Pacific team",
        "Thomson Reuters reporting team",
        "Regulatory Intelligence Team",
        "Reuters Communications Team",
        "By CNN Impact Your World Team",
        "Thomson Reuters Reporting Team",
        "CNBC Prime Team",
        "CNN's Reality Check Team",
        "Kelley Blue Book",
        "POLITICO Magazine",
        "WIRED Readers",
        "PEOPLE style",
        "Axios PR",
        "Vox First Person",
        "NewsFeed with @BuzzFeedBen",
        "Lynx Insight Service",
        "Washington Examiner",
        "Washington Free Beacon",
        "joellegoldsteintimeinc",
        "Insider Inc.",
        "Vox Culture",
        "By CNN's Reality Check Team",
        "Mashable Brand X",
        "Mashable Video",
        "Mashable Studios",
        "Landon Peoples",
        "PeopleStyle",
        "FoxSports",
        "The Wall Street Journal",
        "CNN Library",
        "You",
        "CNN",
        "Us",
        "Refinery29",
        "School of Visual Arts (SVA)",
        "WIRED Staff",
        "Field Level Media",
        "Associated Press",
        "The Associated Press",
        "People Staff",
        "Axios",
        "The Editorial Board",
        "The Hill staff",
        "The Hill Staff",
        "The New York Times",
        "The Learning Network",
        "The New York Times Magazine",
        "Fox News",
        "VICE News",
        "VICE News and Reuters",
        "VICE Sports",
        "PCMag",
        "Newser",
        "Travel + Leisure",
        "AccuWeather",
        "New York Post",
        "news.com.au",
        "The Sun",
        "The Daily Meal",
        "Space.com",
        "The Kim Komando Show",
        "ET Online",
        "Realtor.com",
        "Bill Press",
        "BuzzFeed News",
        "BuzzFeed News Video",
        "Reuters News Agency",
        "IFR News",
        "eBay",
        "Cleveland Institute of Art",
        "SCAD FASH Museum of Fashion + Film",
        "Paris",
        "STRATFOR Worldview",
        "View-Master Deluxe VR Viewer",
        "Dell",
        "Breakingviews columnists",
        "John",
        "Woods",
        "Liz Spayd with Evan Gershkovich",
        "UBS",
        " ",
        "Reuters Communications",
        "By CNN",
        "CNN Business",
        "CNN Business ",
        "Virginia Commonwealth University (VCUarts)",
        "Taipei Fine Arts Museum (TFAM)",
        "BRIC",
        "Motherboard Stuff",
        "Vermont College of Fine Arts (VCFA)",
        "Business Insider Australia",
        "Spark Business®",
        "Business Insider Intelligence",
        " CNN Business",
        "Healthline.com",
        "LifeZette.com",
        "kbender1271",
        "R29 Brand Experiences",
        "Mashable Brand X",
        "Axios Events",
        "Vice Money",
        "Vice Beta",
        "Axios Video",
        "Verge Video",
        "Maine College of Art (MECA)",
        "\xa0",
        "\u2009",
        "\u200b",
        "Booking.com",
        "HP Australia",
        "Lexus IS",
        "University of Connecticut (UConn)",
        "Boston University",
        "University of Phoenix",
        "Shell",
        "Honor",
        "Provided by Comcast NBCUniversal LIFT Labs",
        "CNN Social Discovery Team",
        "Mashable Team",
        "IBM",
        "Vox Conversations",
        "SF Motors",
        "New York Studio School (NYSS)",
        "Center for Maine Contemporary Art",
        "Virginia Museum of Contemporary Art (MOCA)",
        "Southern California Institute of Architecture (SCI-Arc)",
        "College Art Association (CAA)",
        "San Francisco Art Institute (SFAI)",
        "Centre de Cultura Contemporània de Barcelona (CCCB)",
        "Pennsylvania Academy of Fine Arts (PAFA)",
        "August Wilson Cultural Center",
        "Art Students League of New York",
        "Emily Carr University",
        "She Picks Up Pennies",
        "La Vaquita  Negra del Portal",
        "Washington University in St. Louis",
        "University of Albany",
        "George Washington University",
        "University of Indianapolis",
        "McKnight Foundation",
        "Cornell University",
        "Yale University School of Art",
        "Georgia State University",
        "University of Texas at Austin",
        "New York University",
        "Huawai P9 EE",
        "By CNN Opinion",
        "CNN's Media Team",
        "By CNN's Media Team",
        "Canada Keep Exploring",
        "Brought to you byAlliance for Lifetime Income",
        "Insider Entertainment Team",
        "CNN iReport Team",
        "CNN's Social Discovery Team",
        "By CNN's Capitol Hill team",
        "MIT Press",
        "Press Release",
        "PRESS RELEASE",
        "Paytm Canada",
        "Green Point Projects",
        "Project HEAL",
        "Hunter College",
        "Columbia College Chicago",
        "Queens College",
        "SUNY Purchase College",
        "Maryland Institute College of Art",
        "Warren Wilson College",
        "Studio Arts College International",
        "Bard College",
        "Mills College Art Museum",
        "Yale Center For British Art",
        "Philadelphia Museum of Art",
        "Tyler School of Art",
        "Spencer Museum of Art",
        "Clio Art Fair",
        "Association of International Photography Art Dealers",
        "Chrysler Museum of Art",
        "Sotheby's Institute of Art",
        "Nevada Museum of Art",
        "Samek Art Museum",
        "Harwood Museum of Art",
        "New Hampshire Institute of Art",
        "Neuberger Museum of Art",
        "Wellin Museum of Art",
        "Cranbrook Academy of Art",
        "Art in Dumbo",
        "Peabody Essex Museum",
        "Rhode Island School of Design",
        "Thomson Reuters Foundation",
        "Lindsey Byrnes; Words by Allie Conti",
        "Disability/Arts/NYC (DANT)",
        "Katonah Museum of Art",
        "Mural Arts Philadelphia",
        "Laguna Art Museum",
        "Mead Art Museum",
        "A Police Officer's Wife ",
        "KW Institute for Contemporary Art",
        "San Francisco Art Book Fair",
        "Frost Art Museum FIU",
        "Art World Conference",
        "Birmingham Museum of Art",
        "International Art Space",
        "Center for Contemporary Political Art",
        "Santa Barbara Museum of Art",
        "SCAD Museum of Art",
        "Laznia Center For Contemporary Art",
        "Delaware Art Museum",
        "Marlborough Fine Art",
        "Harvard Art Museums",
        "El Paso Museum of Art",
        "Sven Berger as told to Maurice Chammah",
        "Times Square Arts",
        "Hrag VartanianBenjamin Sutton",
        "José Luis Pardo / Alejandra S. Inzunza",
        "Emily Sia Lian Wong",
        "Art on Paper",
        "Pulitzer Arts Foundation",
        "One River School of Art + Design",
        "Lucky's Cafe",
        "Carnegie Museum of Art",
        "Oscar Webb; Words: Yohann Koshy",
        "Various Artists",
        "Voices in Contemporary Art",
        "Bard Graduate Center",
        "Caribbean Cultural Center African Diaspora Institute",
        "International Center of Photography",
        "Skirball Cultural Center",
        "New York City Center",
        "Solomon R. Guggenheim Museum",
        "Center For Reproductive Rights",
        "Eiteljorg Museum",
        "Norman Rockwell Museum",
        "Corning Museum of Glass",
        "George Eastman Museum",
        "Manetti Shrem Museum",
        "Norton Simon Museum",
        "Penn Museum",
        "KW Institute for Contemporary Art",
        "Contemporary And",
        "National Park Foundation",
        " Marriott Rewards Premier Credit Card from Chase",
        "Vodafone Foundation",
        "Northeastern University",
        "University of Notre Dame",
        "University of Houston",
        "American Jewish University",
        "University of Hartford",
        "Colorado State University",
        "City College of New York",
        "Brooklyn College",
        "CSG International ",
        "Institute for Responsible Technology",
        "Transart Institute",
        "China Institute",
        "Cohen Media",
        "By Common Sense Media",
        "Stat News",
        "Reuters et VICE News",
        "New York Botanical Garden",
        "Hilton Garden Inn",
        "Heineken Shape Your City",
        "Industry City",
        "Samsung Mobile Malaysia",
        "Samsung BeFearless",
        "Samsung AU",
        "Samsung SmartThings",
        "Best Buy",
        "Samsung India",
        "Pratt Institute",
        "Brooklyn Arts Council",
        "Manufacturers Village Artists",
        "Newark Arts Council",
        "Unlikely Journal for Creative Arts",
        "New York Times Graphics",
        "Lower Manhattan Cultural Council",
        "Eli's Cheesecake Company",
        "Business Insider News Team",
        "Medill  National Security Reporting Project",
        "Japan National Tourism Organization",
        "International Rescue Committee",
        "National Academy of Design",
        "International Herald Tribune",
        "INTERNATIONAL HERALD TRIBUNE",
        "Bureau of Investigative Journalism",
        "Taste of Country",
        "Deena ElGenaidiZachary SmallJasmine Weber",
        "Frankie Mullin; Illustrations: Polly Williams",
        "92nd Street Y",
        "1 1",
        "1310 GMT shares",
        "2016 Washington Auto Show",
        "Pop Culture Died in 2009",
        "O2 Advertorial",
        "R29 Unbothered",
        "20th Century Fox Home Entertainment",
        "France 24",
        "Central Intelligence",
        "France 24",
        "By 44 European Politicians",
        "Moto G5 Plus",
        "Moto G5 ",
        "7-Eleven Canada ",
        "Las Vegas",
        "Adobe Marketing Cloud",
        'Adobe Experience Cloud',
        "Seph RodneyJillian SteinhauerBenjamin SuttonHrag VartanianClaire Voon",
        "Jordan Crook\t\t\t\n\n\t\t\t\tJonathan Shieber",

    ] + list(set(clean_ds["publication"]))
    BAD_AUTHORS = [
        a.lower() for a in BAD_AUTHORS
    ]

    return BAD_AUTHORS


def filtering_author(author):
    if not isinstance(author, str) or not author.strip():
        return False

    if author.lower() in BAD_AUTHORS:
        return False

    if 'staff' in author.lower() or 'editor' in author.lower() or 'the ' in author.lower() or '\t' in author.lower() or '\n' in author.lower() :
        return False
    if len(author.split())==1:
        return False
    word_list = author.split(",")
    word_list = [word.strip() for word in word_list]
    if len(word_list) > 1:
         if word_list[0].lower() in pub_word_list:
             return False
         if len(word_list) == 2:
            if word_list[1] in ["MD","Motorsport.com","Loan Pricing Corporation","Thomson Reuters Foundation","journalist in Prague", "Kaiser Family Foundation", "Thomson Reuters Foundation","for CNN","global CIO at UBS Wealth Management","CEO of TipRanks","CEO of Dezenhall Resources","a business ethics professor at Notre Dame","M.d.","contributor", " CNN","PHD","PhD","M.D.","Opinion Contributor","CNN",'CNN Business', 'CNN Legal Analyst','CNN National Security Analyst','Kaiser Health News','University of Chicago Institute of Politics','Common Sense Media','opinion contributor', 'CNN', 'Special to CNN', 'CNN Chief National Correspondent', 'University of Chicago Institute of Politics']:
                return True
            else:
                return False
         else:
             return False

    if author =="":
        return False
    # a_list = author.split()
    # for word in a_list:
    #     if word.lower() in pub_word_list:
    #         return False
    
    return True

def sample_texts_from_articles(article_texts, max_texts=16):
    articles = []
    all_sentences = set()

    for text in article_texts:
        if not isinstance(text, str) or not text.strip():
            continue

        sents = list(dict.fromkeys(sent_tokenize(text)))
        if sents:
            articles.append(sents)
            all_sentences.update(sents)

    if not articles:
        return [], False

    if len(articles) >= max_texts:
        chosen_articles = random.sample(articles, max_texts)
        selected = [random.choice(sents) for sents in chosen_articles]
        return selected, True
    else:

        all_sentences = list(all_sentences)

        selected = []
        used = set()

        n = len(articles)

        quotas = [max_texts // n] * n
        remainder = max_texts - sum(quotas)

        for i in random.sample(range(n), min(remainder, n)):
            quotas[i] += 1

        for sents, q in zip(articles, quotas):

            candidates = [s for s in sents if s not in used]
            take = min(q, len(candidates), max_texts - len(selected))

            chosen = random.sample(candidates, take)
            selected.extend(chosen)
            used.update(chosen)

        if len(selected) < max_texts:
            remaining = [s for s in all_sentences if s not in used]
            need = max_texts - len(selected)
            selected.extend(random.sample(remaining, min(need, len(remaining))))

        sufficient = len(selected) >= max_texts

        return selected, sufficient



def main():
    args = parse_args()
    ds = load_source(args)

    # Original curation, retaining its operation order and insertion order.
    author_counts = Counter(a for a in ds["train"]["author"] if a is not None)
    clean_ds = ds["train"].filter(lambda ex: keep_author(ex["author"]))
    clean_ds = clean_ds.filter(lambda ex: keep_article(ex["article"]))
    publication_counts = Counter(clean_ds["publication"])
    global BAD_AUTHORS
    BAD_AUTHORS = build_bad_authors(clean_ds)
    global pub_word_list
    pub_word_list = []
    for pub in list(publication_counts.keys()):
        if pub is not None:
            a = pub.split()
            a = [word.lower() for word in a]
            pub_word_list += a
    pub_word_list = list(set(pub_word_list))
    test_clean_ds = clean_ds.filter(lambda ex: filtering_author(ex["author"]))

    # Original author split. Do not reset random before sentence sampling.
    author_counts = Counter(test_clean_ds["author"])
    authors = list(author_counts.keys())
    random.seed(42)
    random.shuffle(authors)
    n = len(authors)
    n_train = int(0.9 * n)
    n_valid = int(0.05 * n)
    train_authors = set(authors[:n_train])
    valid_authors = set(authors[n_train:n_train + n_valid])
    if args.author_list_dir is not None:
        # Published lists (reproduces the paper's split). The shuffle above is
        # kept so the random state for sentence sampling is unchanged.
        with open(args.author_list_dir / "train_authors.json") as f:
            train_authors = set(json.load(f))
        with open(args.author_list_dir / "valid_authors.json") as f:
            valid_authors = set(json.load(f))

    train_ds = test_clean_ds.filter(lambda ex: ex["author"] in train_authors)
    valid_ds = test_clean_ds.filter(lambda ex: ex["author"] in valid_authors)
    article_splits = DatasetDict({"train": train_ds, "valid": valid_ds})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    article_output_path = args.output_dir / "one_author_article_dataset_split"
    if not article_output_path.exists():
        article_splits.save_to_disk(str(article_output_path))
    if args.skip_sentence_selection:
        return

    ensure_nltk_data()

    def filtering(new_ds):
        # This is equivalent to the original index-based loop above, but reads
        # the two Arrow columns once. It preserves article order and every call
        # to random.sample/random.choice, while avoiding millions of scalar reads.
        author_to_articles = defaultdict(list)
        for author, article in zip(new_ds["author"], new_ds["article"]):
            author_to_articles[author].append(article)
        author_to_16_texts = {}
        for author, articles in tqdm(author_to_articles.items()):
            texts, is_fill_16 = sample_texts_from_articles(articles, max_texts=16)
            if is_fill_16:
                author_to_16_texts[author] = texts
        return author_to_16_texts

    train_filtered_texts = filtering(train_ds)
    valid_filtered_texts = filtering(valid_ds)

    def flatten_author_texts(filtered_data):
        rows = []
        for author, texts in filtered_data.items():
            for text in texts:
                rows.append({"author": author, "text": text})
        return rows

    filtered_ds_dict = DatasetDict({
        "train": Dataset.from_list(flatten_author_texts(train_filtered_texts)),
        "valid": Dataset.from_list(flatten_author_texts(valid_filtered_texts)),
    })
    filtered_ds_dict.save_to_disk(str(args.output_dir / "one_author_article_dataset_split_filtered"))


if __name__ == "__main__":
    main()
