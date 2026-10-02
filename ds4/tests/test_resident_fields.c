/* field_u64 coverage: strict JSON unsigned-integer parsing for
 * spec_run_sampled's seed.  Includes the resident source with its main
 * compiled out and links the stub engine. */
#define DS4_RESIDENT_TEST
#include "../../harness/protocol-adapter/ds4_shim/ds4_resident.c"

static int failures;
#define CHECK(cond, name) do { \
    if (!(cond)) { failures++; \
        fprintf(stderr, "FAIL %s\n", name); } \
    else printf("ok %s\n", name); } while (0)

static int try_seed(const char *json, uint64_t *out) {
    return field_u64(json, "seed", out);
}

int main(void) {
    uint64_t v = 0;
    /* Valid: full u64 domain is preserved. */
    CHECK(try_seed("{\"seed\":0}", &v) == 0 && v == 0, "seed 0");
    CHECK(try_seed("{\"seed\":1}", &v) == 0 && v == 1, "seed 1");
    CHECK(try_seed("{\"seed\":9223372036854775807}", &v) == 0 &&
          v == 9223372036854775807ull, "seed LLONG_MAX");
    CHECK(try_seed("{\"seed\":9223372036854775808}", &v) == 0 &&
          v == 9223372036854775808ull, "seed LLONG_MAX+1");
    CHECK(try_seed("{\"seed\":18446744073709551615}", &v) == 0 &&
          v == 18446744073709551615ull, "seed UINT64_MAX");
    /* Whitespace and delimiters that are legal around a value. */
    CHECK(try_seed("{\"seed\": 42 }", &v) == 0 && v == 42, "ws seed");
    CHECK(try_seed("{\"count\":4,\"seed\":7,\"first_token\":1}", &v) == 0 &&
          v == 7, "seed mid-object");
    CHECK(try_seed("{\"seed\":\t13\t}", &v) == 0 && v == 13, "tab ws");
    CHECK(try_seed("{\"seed\":5 ,\"x\":1}", &v) == 0 && v == 5,
          "ws before comma");
    /* Invalid: nothing may sneak past. */
    CHECK(try_seed("{\"seed\":18446744073709551616}", &v) == -1,
          "seed 2^64 overflow");
    CHECK(try_seed("{\"seed\":-1}", &v) == -1, "negative seed");
    CHECK(try_seed("{\"seed\":\n-1}", &v) == -1, "newline-negative");
    CHECK(try_seed("{\"seed\":+1}", &v) == -1, "plus sign");
    CHECK(try_seed("{\"seed\":01}", &v) == -1, "leading zero");
    CHECK(try_seed("{\"seed\":1.0}", &v) == -1, "float");
    CHECK(try_seed("{\"seed\":1e3}", &v) == -1, "exponent");
    CHECK(try_seed("{\"seed\":\"5\"}", &v) == -1, "quoted");
    CHECK(try_seed("{\"seed\":1 garbage}", &v) == -1, "trailing junk");
    CHECK(try_seed("{\"seed\":1\tfoo}", &v) == -1, "ws then junk");
    CHECK(try_seed("{\"seed\":}", &v) == -1, "empty value");
    CHECK(try_seed("{\"seed\":x}", &v) == -1, "non-digit");
    CHECK(try_seed("{\"other\":1}", &v) == -1, "missing key");
    if (failures) {
        fprintf(stderr, "resident field tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("resident field tests: ok");
    return 0;
}
