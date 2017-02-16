Praats/external/flac/READ_ME.TXT
Paul Boersma, 18 December 2014
This file describes the adaptations to the FLAC 1.2.1 sources
that are needed to make them compatible with Praat.

The .c and .h files are put into the single FLAC directory.

The #include statements are flattened, e.g.
#include private/float.h becomes #include flac_private_float.h.

The FLaC__INLINE statement is turned into an inline statement
in flac_share_alloc.h, and removed elsewhere (i.e. wherever there is a
compiler message).

For MinGW we need to change in flac_share_alloc.h:

#if !defined _MSC_VER && !defined __EMX__
#include <stdint.h> /* for SIZE_MAX in case limits.h didn't get it */
#endif

The sources contain a confusion of FLAC__int32 and int,
especially in calls to local_bitreader_read_rice_signed_block
or FLAC__bitreader_read_rice_signed_block;
Some changes from int to FLAC__int32 may be necessary.
We also have to insert
Melder_assert (sizeof (int) == 4)
in read_residual_partitioned_rice_.

To ensure compatibility with international file names on Windows,
the following is added to flac_FLAC_formant.h:
#ifdef _WIN32
	#include "melder.h"
	#define fopen(filename,mode)  _wfopen (Melder_peekWcsToUtf16 (Melder_peekUtf8ToWcs (filename)), L"" mode)
#endif

There were two mistakes in FLAC__MD%Final () in flac_md5.c:
	memset(ctx, 0, sizeof(ctx));	/* In case it's sensitive */
	if(0 != ctx->internal_buf) {
		free(ctx->internal_buf);
		ctx->internal_buf = 0;
		ctx->capacity = 0;
	}
should be
	if(0 != ctx->internal_buf) {   // test before clearing!
		free(ctx->internal_buf);
		ctx->internal_buf = 0;
		ctx->capacity = 0;
	}
	memset(ctx, 0, sizeof(*ctx));	// clear the whole structure, not just the first four or eight bytes!

int16_t * Melder_peekWcsToUtf16 (const wchar_t *string);
#endif

>> voices.cpp : voice_t *LoadVoice(const char *vname, int control)

#ifdef DATA_FROM_SOURCECODE_FILES
	long numberOfBytes;
	const char * data;
	if (tone_only) {
		data = espeakdata_get_voiceVariant (vname, &numberOfBytes);
	} else {
		language_type = "en";
		data = espeakdata_get_voice (vname, &numberOfBytes);
		language_type = vname;
	}
	if (data == 0) {
		language_type = "en";    // default
		data = espeakdata_get_voice ("en/en", &numberOfBytes);
	}
#else
...
endif

#ifdef DATA_FROM_SOURCECODE_FILES
	long index = 1;
	const char *start = data;
	while (start = espeakdata_get_voicedata (start, numberOfBytes, buf, sizeof(buf), &index)) {
#else
	while((f_voice != NULL) && (fgets_strip(buf,sizeof(buf),f_voice) != NULL)) {
#endif

>> synthdata : int LoadPhData()

#ifdef DATA_FROM_SOURCECODE_FILES
	long llength;
	phoneme_tab_data = (unsigned char *) FilesInMemory_getData (espeakdata_phons, L"phontab", &llength);
	phoneme_index = (USHORT *) FilesInMemory_getData (espeakdata_phons, L"phonindex", &llength);
	phondata_ptr = (char *) FilesInMemory_getData (espeakdata_phons, L"phondata", &llength);
	tunes = (TUNE *) FilesInMemory_getData (espeakdata_phons, L"intonations", &llength);
	length = llength;
#else
	if((phoneme_tab_data = (unsigned char *)ReadPhFile((void *)(phoneme_tab_data),"phontab",NULL)) == NULL)
		return(-1);
	if((phoneme_index = (USHORT *)ReadPhFile((void *)(phoneme_index),"phonindex",NULL)) == NULL)
		return(-1);
	if((phondata_ptr = ReadPhFile((void *)(phondata_ptr),"phondata",NULL)) == NULL)
		return(-1);
	if((tunes = (TUNE *)ReadPhFile((void *)(tunes),"intonations",&length)) == NULL)
		return(-1);
#endif

void FreePhData(void)
{//==================
#ifndef DATA_FROM_SOURCECODE_FILES
	Free(phoneme_tab_data);
	Free(phoneme_index);
	Free(phondata_ptr);
	Free(tunes);
#endif
	phoneme_tab_data=NULL;
	phoneme_index=NULL;
	phondata_ptr=NULL;
	tunes=NULL;
}

>> speak_lib.cpp


#ifdef DATA_FROM_SOURCECODE_FILES
static void init_path(const char *path) {
	(void) path;
}
#else
static void init_path(const char *path)
{//====================================
#ifdef PLATFORM_WINDOWS

near line 115 and 465: sleep is not a WIN64 function, but luckily we don't need it.
// sleep(1)


>> dictionary.cpp

#ifdef DATA_FROM_SOURCECODE_FILES
int LoadDictionary(Translator *tr, const char *name, int no_error)
{
	strcpy (dictionary_name, name);   // currently loaded dictionary name
	strcpy (tr -> dictionary_name, name);

	// Load a pronunciation data file into memory
	// bytes 0-3:  offset to rules data
	// bytes 4-7:  number of hash table entries

	if(tr -> data_dictlist != NULL) {
		Free (tr -> data_dictlist);
		tr -> data_dictlist = NULL;
	}
	unsigned int size;
	tr -> data_dictlist = (char *) espeakdata_get_dict_data (name, &size);
	if (tr -> data_dictlist == 0) {
		return 1;
	}
	int *pw = reinterpret_cast<int *> (tr -> data_dictlist);
	int length = Reverse4Bytes (pw[1]); // was int really written with 4 bytes?

	if (size <= (N_HASH_DICT + sizeof(int)*2)) {
		Melder_error_ (L"Empty _dict: ", Melder_utf8ToWcs(name), L"_dict.");
		return(2);
	}

	if((Reverse4Bytes(pw[0]) != N_HASH_DICT) || (length <= 0) || (length > 0x8000000)) {
		Melder_error_ (L"Bad data in dict: ", Melder_utf8ToWcs(name), L" ", Melder_integer (Reverse4Bytes(pw[0])), L" ", Melder_integer (length));
		return (2);
	}
	tr -> data_dictrules = &(tr->data_dictlist[length]);

	// set up indices into data_dictrules
	InitGroups(tr);
	if (tr -> groups1[0] == NULL) {
		Melder_error_ (L"Error in ", Melder_peekUtf8ToWcs (name), L"_rules, no default rule group.");
	}

	// set up hash table for data_dictlist
	char *p = &(tr -> data_dictlist[8]);

	for (int hash = 0; hash < N_HASH_DICT; hash++) {
		tr -> dict_hashtab[hash] = p;
		while ((length = *p) != 0) {
			p += length;
		}
		p++;   // skip over the zero which terminates the list for this hash value
	}

	return (0);
}  //  end of LoadDictionary

#else
int LoadDictionary_old(Translator *tr, const char *name, int no_error)


************ klatt.cpp  ***************

replace the gen_noise with the corrected version:
static double gen_noise(double noisedummy) // repaired ppgb 20111223
{
        long temp;
        static double nlast = 0.0;

        temp = (long) getrandom (-8191, 8191);
        kt_globals.nrand = (long) temp;

        double noise = kt_globals.nrand + (0.75 * nlast);
        nlast = noise;

        return(noise);
}

//in speech.h
template <size_t _align_to, typename T> T *align_address (T *p)
{
	union {
		T* ptr;
		size_t integer;
	};
	const size_t bit_mask = ~(_align_to - 1);
	ptr = p;
	Melder_assert (sizeof (size_t) == sizeof (void *));
	integer &= bit_mask;
	return ptr;
}

// tr_languages
replace 
const char string_ordinal[] = {0xc2,0xba,0};  // masculine ordinal character, UTF-8
by
const char string_ordinal[] = {static_cast<char>(0xc2),static_cast<char>(0xba),0};  // masculine ordinal character, UTF-8
to compile on 32 bit systems

20161101
voices.cpp: increase size of array voice_identifier[40] to prevent buffer overflow static char voice_identifier[80];
