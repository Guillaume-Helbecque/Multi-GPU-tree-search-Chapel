module Taillard
{
  const time_seeds: [1..120] int(64) =
  [   873654221 /*ta001*/,    379008056 /*ta002*/,  1866992158 /*ta003*/, 216771124 /*ta004*/,  495070989 /*ta005*/,
      402959317 /*ta006*/,  1369363414 /*ta007*/, 2021925980 /*ta008*/, 573109518 /*ta009*/,  88325120 /*ta010*/,
      587595453 /*ta011*/,  1401007982 /*ta012*/, 873136276 /*ta013*/,  268827376 /*ta014*/,  1634173168 /*ta015*/,
      691823909 /*ta016*/,  73807235 /*ta017*/,   1273398721 /*ta018*/, 2065119309 /*ta019*/, 1672900551 /*ta020*/,
      479340445 /*ta021*/,  268827376 /*ta022*/,  1958948863 /*ta023*/, 918272953 /*ta024*/,  555010963 /*ta025*/,
      2010851491 /*ta026*/, 1519833303 /*ta027*/, 1748670931 /*ta028*/, 1923497586 /*ta029*/, 1829909967 /*ta030*/,
      1328042058 /*ta031*/, 200382020 /*ta032*/,  496319842 /*ta033*/,  1203030903 /*ta034*/, 1730708564 /*ta035*/,
      450926852 /*ta036*/,  1303135678 /*ta037*/, 1273398721 /*ta038*/, 587288402 /*ta039*/,  248421594 /*ta040*/,
      1958948863 /*ta041*/, 575633267 /*ta042*/,  655816003 /*ta043*/,  1977864101 /*ta044*/, 93805469 /*ta045*/,
      1803345551 /*ta046*/, 49612559 /*ta047*/,   1899802599 /*ta048*/, 2013025619 /*ta049*/, 578962478 /*ta050*/,
      1539989115 /*ta051*/, 691823909 /*ta052*/,  655816003 /*ta053*/,  1315102446 /*ta054*/, 1949668355 /*ta055*/,
      1923497586 /*ta056*/, 1805594913 /*ta057*/, 1861070898 /*ta058*/, 715643788 /*ta059*/,  464843328 /*ta060*/,
      896678084 /*ta061*/,  1179439976 /*ta062*/, 1122278347 /*ta063*/, 416756875 /*ta064*/,  267829958 /*ta065*/,
      1835213917 /*ta066*/, 1328833962 /*ta067*/, 1418570761 /*ta068*/, 161033112 /*ta069*/,  304212574 /*ta070*/,
      1539989115 /*ta071*/, 655816003 /*ta072*/,  960914243 /*ta073*/,  1915696806 /*ta074*/, 2013025619 /*ta075*/,
      1168140026 /*ta076*/, 1923497586 /*ta077*/, 167698528 /*ta078*/,  1528387973 /*ta079*/, 993794175 /*ta080*/,
      450926852 /*ta081*/,  1462772409 /*ta082*/, 1021685265 /*ta083*/, 83696007 /*ta084*/,   508154254 /*ta085*/,
      1861070898 /*ta086*/, 26482542 /*ta087*/,   444956424 /*ta088*/,  2115448041 /*ta089*/, 118254244 /*ta090*/,
      471503978 /*ta091*/,  1215892992 /*ta092*/, 135346136 /*ta093*/,  1602504050 /*ta094*/, 160037322 /*ta095*/,
      551454346 /*ta096*/,  519485142 /*ta097*/,  383947510 /*ta098*/,  1968171878 /*ta099*/, 540872513 /*ta100*/,
      2013025619 /*ta101*/, 475051709 /*ta102*/,  914834335 /*ta103*/,  810642687 /*ta104*/,  1019331795 /*ta105*/,
      2056065863 /*ta106*/, 1342855162 /*ta107*/, 1325809384 /*ta108*/, 1988803007 /*ta109*/, 765656702 /*ta110*/,
      1368624604 /*ta111*/, 450181436 /*ta112*/,  1927888393 /*ta113*/, 1759567256 /*ta114*/, 606425239 /*ta115*/,
      19268348 /*ta116*/,   1298201670 /*ta117*/, 2041736264 /*ta118*/, 379756761 /*ta119*/,  28837162 /*ta120*/ ];

  proc taillard_get_nb_jobs(const id: int): int(32)
  {
    if (id > 110) then return 500;
    if (id > 90) then return 200;
    if (id > 60) then return 100;
    if (id > 30) then return 50;
    /*if(id>0)*/ return 20;
  }

  proc taillard_get_nb_machines(const id: int): int(32)
  {
    if (id > 110) then return 20; //500x20
    if (id > 100) then return 20; //200x20
    if (id > 90) then return 10; //200x10
    if (id > 80) then return 20; //100x20
    if (id > 70) then return 10; //100x10
    if (id > 60) then return 5;  //100x5
    if (id > 50) then return 20; //50x20
    if (id > 40) then return 10; //50x10
    if (id > 30) then return 5;  //50x5
    if (id > 20) then return 20; //20x20
    if (id > 10) then return 10; //20x10
    /*if(id>0 )*/ return 5; //20x5
  }

  proc taillard_get_best_ub(const id: int): int
  {
    const optimal: [1..120] int = [1278, 1359, 1081, 1293, 1235, 1195, 1234, 1206, 1230, 1108,            // 20x5
                                 1582, 1659, 1496, 1377, 1419, 1397, 1484, 1538, 1593, 1591,            // 20x10
                                 2297, 2099, 2326, 2223, 2291, 2226, 2273, 2200, 2237, 2178,            // 20x20
                                 2724, 2834, 2621, 2751, 2863, 2829, 2725, 2683, 2552, 2782,            // 50x5
                                 2991, 2867, 2839, 3063, 2976, 3006, 3093, 3037, 2897, 3065,            // 50x10
                                 3846, 3699, 3640, 3719, 3610, 3679, 3704, 3691, 3741, 3755,            // 50x20
                                 5493, 5268, 5175, 5014, 5250, 5135, 5246, 5094, 5448, 5322,            // 100x5
                                 5770, 5349, 5676, 5781, 5467, 5303, 5595, 5617, 5871, 5845,            // 100x10
                                 6202, 6183, 6271, 6269, 6314, 6364, 6268, 6401, 6275, 6434,            // 100x20
                                 10862, 10480, 10922, 10889, 10524, 10329, 10854, 10730, 10438, 10675,  // 200x10
                                 11195, 11203, 11281, 11275, 11259, 11176, 11360, 11334, 11192, 11284,  // 200x20
                                 26040, 26520, 26371, 26456, 26334, 26477, 26389, 26560, 26005, 26457]; // 500x20

    return optimal[id];
  }

  private proc unif(ref seed: int(64), low: int(64), high: int(64)): int(64)
  {
    var m: int(64) = 2147483647, a: int(64) = 16807, b: int(64) = 127773, c: int(64) = 2836, k: int(64);
    var value_0_1: real;

    k = seed / b;
    seed = a * (seed % b) - k * c;
    if (seed < 0) then
      seed = seed + m;
    value_0_1 = seed:real / m:real;

    return low + (value_0_1 * (high - low + 1)): int(64);
  }

  proc taillard_get_processing_times(ref ptm: [] int(32), const id: int): void
  {
    const N = taillard_get_nb_jobs(id);
    const M = taillard_get_nb_machines(id);
    var time_seed = time_seeds[id];

    for i in 0..#M {
      for j in 0..#N {
        ptm[i*N+j] = unif(time_seed, 1, 99):int(32);
      }
    }
  }
}
