import csv
import json
import time
import requests

USER_AGENT = "Hermes/1.0 (Universal Intelligence Platform)"

SEED_ARTISTS = [
    "Eminem", "Taylor Swift", "Drake", "The Beatles", "Radiohead",
    "Kendrick Lamar", "Beyonce", "Ed Sheeran", "Adele", "Coldplay",
    "Rihanna", "Kanye West", "Billie Eilish", "The Weeknd", "Ariana Grande",
    "Post Malone", "Travis Scott", "Dua Lipa", "Bruno Mars", "Lady Gaga",
    "Michael Jackson", "Elvis Presley", "Madonna", "Prince", "David Bowie",
    "Led Zeppelin", "Pink Floyd", "Queen", "Nirvana", "AC/DC",
    "Bob Marley", "Stevie Wonder", "Whitney Houston", "Elton John", "Bob Dylan",
    "Jay-Z", "Lil Wayne", "50 Cent", "Snoop Dogg", "Tupac Shakur",
    "Notorious B.I.G.", "Nas", "J. Cole", "Future", "Migos",
    "Arctic Monkeys", "Tame Impala", "Imagine Dragons", "Linkin Park", "Green Day",
    "Red Hot Chili Peppers", "Foo Fighters", "U2", "The Rolling Stones", "Aerosmith",
    "Metallica", "Iron Maiden", "Black Sabbath", "Guns N' Roses", "Def Leppard",
    "Daft Punk", "Deadmau5", "Skrillex", "Marshmello", "Calvin Harris",
    "Sia", "Lorde", "Halsey", "Doja Cat", "Cardi B",
    "Bad Bunny", "J Balvin", "Shakira", "Daddy Yankee", "Ozuna",
    "BTS", "BLACKPINK", "Stray Kids", "TWICE", "EXO",
    "Arijit Singh", "A.R. Rahman", "Shreya Ghoshal", "Atif Aslam", "Pritam",
    "Mozart", "Beethoven", "Bach", "Chopin", "Vivaldi",
    "Miles Davis", "John Coltrane", "Charlie Parker", "Louis Armstrong", "Duke Ellington",
    "Frank Sinatra", "Nat King Cole", "Ray Charles", "Aretha Franklin", "Nina Simone",
]

SEED_BOOKS = [
    "Dune", "Harry Potter", "The Great Gatsby", "1984", "To Kill a Mockingbird",
    "Pride and Prejudice", "The Catcher in the Rye", "Lord of the Rings", "The Hobbit", "Brave New World",
    "Animal Farm", "Fahrenheit 451", "Sapiens", "Thinking Fast and Slow", "The Art of War",
    "Crime and Punishment", "War and Peace", "Don Quixote", "Moby Dick", "Dracula",
    "Frankenstein", "Jane Eyre", "Wuthering Heights", "Great Expectations", "A Tale of Two Cities",
    "The Odyssey", "The Iliad", "Hamlet", "Macbeth", "Romeo and Juliet",
    "One Hundred Years of Solitude", "Love in the Time of Cholera", "The Alchemist", "Siddhartha", "Steppenwolf",
    "The Name of the Rose", "The Handmaid's Tale", "Catch-22", "Slaughterhouse-Five", "The Road",
    "Gone Girl", "The Girl with the Dragon Tattoo", "The Da Vinci Code", "Angels and Demons", "Inferno",
    "A Brief History of Time", "The Selfish Gene", "The Origin of Species", "Silent Spring", "Cosmos",
    "Atomic Habits", "Deep Work", "The Lean Startup", "Zero to One", "Rich Dad Poor Dad",
    "Educated", "Becoming", "Born a Crime", "Long Walk to Freedom", "I Know Why the Caged Bird Sings",
    "The Kite Runner", "A Thousand Splendid Suns", "Life of Pi", "The Book Thief", "The Fault in Our Stars",
    "Twilight", "The Hunger Games", "Divergent", "Ender's Game", "Ready Player One",
    "Neuromancer", "Snow Crash", "The Hitchhiker's Guide to the Galaxy", "Foundation", "Dune Messiah",
    "The Stand", "It", "The Shining", "Carrie", "Misery",
    "Percy Jackson", "Eragon", "The Chronicles of Narnia", "His Dark Materials", "A Song of Ice and Fire",
    "The Witcher", "Mistborn", "The Way of Kings", "The Kingkiller Chronicle", "Wheel of Time",
    "The Diary of a Young Girl", "Night", "Man's Search for Meaning", "The Prince", "Meditations",
    "The Republic", "Thus Spoke Zarathustra", "Beyond Good and Evil", "The Social Contract", "Wealth of Nations",
]


def fetch_music_catalog():
    rows = []
    for artist_name in SEED_ARTISTS:
        url = f"https://musicbrainz.org/ws/2/artist/?query={requests.utils.quote(artist_name)}&fmt=json&limit=1"
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            artists = data.get("artists", [])
            if not artists:
                print(f"  No results for: {artist_name}")
                continue
            a = artists[0]
            name = a.get("name", artist_name)
            entity_type = a.get("type", "")
            country = a.get("country", "")
            disambiguation = a.get("disambiguation", "")
            life_span = a.get("life-span", {})
            born = life_span.get("begin", "")
            tags = a.get("tags", [])
            genres = [t.get("name", "") for t in sorted(tags, key=lambda t: t.get("count", 0), reverse=True) if t.get("count", 0) >= 0][:5]
            genres_str = "|".join(genres)
            aliases_raw = a.get("aliases", [])
            aliases = [al.get("name", "") for al in aliases_raw if al.get("name")][:3]
            aliases_str = "|".join(aliases)
            comb = f"{name} {entity_type} {country} {' '.join(genres)} {disambiguation}"
            rows.append({
                "artist_name": name,
                "entity_type": entity_type,
                "country": country,
                "genres": genres_str,
                "born": born,
                "disambiguation": disambiguation,
                "aliases": aliases_str,
                "comb": comb,
            })
            print(f"  OK: {name} ({country}, {genres_str})")
        except Exception as e:
            print(f"  ERROR for {artist_name}: {e}")
        time.sleep(1.2)

    with open("datasets/music_catalog.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["artist_name", "entity_type", "country", "genres", "born", "disambiguation", "aliases", "comb"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nMusic catalog written: {len(rows)} artists")


def fetch_books_catalog():
    rows = []
    for book_title in SEED_BOOKS:
        encoded = requests.utils.quote(book_title)
        url = f"https://openlibrary.org/search.json?q={encoded}&fields=key,title,author_name,first_publish_year,subject,cover_i&limit=1"
        headers = {"User-Agent": USER_AGENT}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("docs", [])
            if not docs:
                print(f"  No results for: {book_title}")
                continue
            d = docs[0]
            title = d.get("title", book_title)
            authors = d.get("author_name", [])
            author_str = "|".join(authors[:3])
            year = d.get("first_publish_year", "")
            subjects_raw = d.get("subject", [])
            subjects = [s for s in subjects_raw if s and not s.startswith("nyt:") and not s.startswith("award:")][:8]
            subjects_str = "|".join(subjects)
            cover_id = d.get("cover_i", "")
            key = d.get("key", "")
            comb = f"{title} {' '.join(authors[:3])} {' '.join(subjects[:5])}"
            rows.append({
                "book_title": title,
                "authors": author_str,
                "first_publish_year": str(year) if year else "",
                "subjects": subjects_str,
                "cover_id": str(cover_id) if cover_id else "",
                "key": key,
                "comb": comb,
            })
            print(f"  OK: {title} by {author_str} ({year})")
        except Exception as e:
            print(f"  ERROR for {book_title}: {e}")
        time.sleep(1.2)

    with open("datasets/books_catalog.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["book_title", "authors", "first_publish_year", "subjects", "cover_id", "key", "comb"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nBooks catalog written: {len(rows)} books")


if __name__ == "__main__":
    print("=== Building Music Catalog ===")
    fetch_music_catalog()
    print("\n=== Building Books Catalog ===")
    fetch_books_catalog()
    print("\n=== Done ===")
