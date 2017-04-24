from collections import OrderedDict, Counter
import numpy as np

def steam(sentence):
    # Do a bit of steaming
    sentence = sentence.replace('.', '').replace(',', '').replace(';', '').replace(':', '').replace('\'', '').replace \
        ('"', '')
    sentence = sentence.lower().split()
    return sentence

def get_vocab(data, nb_words=50000, min_nb=10, remove_stop_words = True):
    """
    Get the vocabulary and the mapping (int to string) of the captions
    :param data:
    :param nb_words:
    :param min_nb:
    :param remove_stop_words:
    :return:
    """


    # Put everything into onw long string
    data = [item for sublist in list(data.values()) for item in sublist]
    data = " ".join(data)

    # Do a bit of steaming
    data = steam(data)
    vocab = Counter(data)

    # Remove the stop words
    new_vocab = vocab.copy()
    for key, value in vocab.items():
        if remove_stop_words and key in stopwords:
            del new_vocab[key]
        if value < min_nb:
            del new_vocab[key]

    vocab = new_vocab

    # Keep the most common words
    vocab = Counter(dict(vocab.most_common(nb_words)))

    # Extract a mapping
    mapping = {}
    mapping[1] = "--UNK--"
    mapping["--UNK--"] = 1
    for i, word in enumerate(sorted(vocab.keys())):
        mapping[i + 2] = word
        mapping[word] = i + 2

    return vocab, mapping


def filter_caps(data, mapping, switch=False):

    """
    Filter the data (remove unknown words, switch to ints, etc...)
    :param data:
    :param mapping:
    :param switch:
    :return:
    """

    data_filtered = {}
    for i, img_name in enumerate(data):
        tmp = []
        for cap in data[img_name]:
            words = steam(cap)
            if switch:
                filtered = [mapping[word] if word in mapping else mapping["--UNK--"] for word in words]
            else:
                filtered = [word if word in mapping else "--UNK--" for word in words]

            tmp.append(filtered)

        data_filtered[img_name] = tmp

    return data_filtered

def pad_to_the_max(data):

    max_len = max([len(x) for x in data])
    data = [x + [0] * (max_len - len(x)) for x in data]
    return np.array(data)


stopwords ="""a
about
above
after
again
against
all
am
an
and
any
are
aren't
as
at
be
because
been
before
being
below
between
both
but
by
can't
cannot
could
couldn't
did
didn't
do
does
doesn't
doing
don't
down
during
each
few
for
from
further
had
hadn't
has
hasn't
have
haven't
having
he
he'd
he'll
he's
her
here
here's
hers
herself
him
himself
his
how
how's
i
i'd
i'll
i'm
i've
if
in
into
is
isn't
it
it's
its
itself
let's
me
more
most
mustn't
my
myself
no
nor
not
of
off
on
once
only
or
other
ought
our
ours
ourselves
out
over
own
same
shan't
she
she'd
she'll
she's
should
shouldn't
so
some
such
than
that
that's
the
their
theirs
them
themselves
then
there
there's
these
they
they'd
they'll
they're
they've
this
those
through
to
too
under
until
up
very
was
wasn't
we
we'd
we'll
we're
we've
were
weren't
what
what's
when
when's
where
where's
which
while
who
who's
whom
why
why's
with
won't
would
wouldn't
you
you'd
you'll
you're
you've
your
yours
yourself
yourselves""".split("\n")