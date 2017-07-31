import unittest
import myres
import json

class TestQuery(unittest.TestCase):
    def testEmpty(self):
        q = myres.myquery()
        self.assertEqual(len(q.url_list),0)
        self.assertEqual(q.count, 0)
        self.assertEqual(q.threshold,0.4)
        q.set_threshold(0.5)
        self.assertEqual(q.threshold,0.5)

    def testLoadfile(self):
        filename = "no_such_file.txt"
        q = myres.myquery(filename)
        self.assertEqual(len(q.url_list),0)
        self.assertEqual(q.count, 0)
        self.assertEqual(q.threshold,0.4)

        filename = "imagelist.txt"
        fp = open(filename)
        lines = fp.readlines()
        fp.close()
        line_num = len(lines)
        q = myres.myquery(filename)
        self.assertEqual(len(q.url_list),line_num)
        self.assertEqual(q.count, line_num)
        self.assertEqual(q.threshold,0.4)

    def testJson(self):
        q = myres.myquery()
        j1 = json.dumps(q, cls=myres.ComplexEncoder)
        q2 = myres.myquery.fromJSON(j1)
        self.assertEqual(q.url_list==q2.url_list, True)
        filename = "imagelist.txt"
        q = myres.myquery(filename)
        q.set_threshold(0.5)
        j1 = json.dumps(q, cls=myres.ComplexEncoder)
        q2 = myres.myquery.fromJSON(j1)
        self.assertEqual(q.url_list==q2.url_list, True)
        self.assertEqual(q.count, q2.count)
        self.assertEqual(q.threshold,q2.threshold)

class TestClassify(unittest.TestCase):
    def testempty(self):
        search  = myres.result_list()
        threshold = 0.3
        self.assertEqual(len(search.results),0)
        res = myres.group_results(threshold)
        self.assertEqual(res.url,'')
        self.assertEqual(len(res.results),0)
        self.assertEqual(res.valid,False)
        self.assertEqual(res.conf_threshold, threshold)
        res.set_url('trial_url')
        self.assertEqual(res.url,'trial_url')
        self.assertEqual(res.conf_threshold, 0.5)
        search.add(res)
        self.assertEqual(len(search.results),1)

        res2 = myres.group_results(1.5)
        self.assertEqual(res2.conf_threshold, 0.5)
        search.add(res2)
        self.assertEqual(len(search.results),2)

    def testURL(self):
        res = myres.group_results(0.2)
        ### run empty url classify
        res.run_classify()
        self.assertEqual(res.results,"invalid URL")
        self.assertEqual(res.conf_threshold, 0.2)
        invalid_link = "invalid_url"
        res.set_url(invalid_link)
        self.assertEqual(res.url,invalid_link)
        self.assertEqual(res.conf_threshold, 0.5)
        res.run_classify()
        self.assertEqual(res.results,"invalid URL")
        res.run_classify_exhaust()
        self.assertEqual(res.results,"invalid URL")
        fake_link = "http://unreal_url.com/no_image.jpg"
        res.set_url(fake_link)
        self.assertEqual(res.url,fake_link)
        res.run_classify()
        self.assertEqual(res.results,"cannot download image.")
        res.run_classify_exhaust()
        self.assertEqual(res.results,"cannot download image.")

        valid_link = "https://cdn1.droom.in/photos/images/drm/super-cars.png"
        res.set_url(valid_link)
        res.run_classify()
        self.assertEqual(len(res.results.results),1)
        self.assertEqual(res.results.results[0].name, 'sports')
        self.assertEqual(res.results.results[0].location, [0,0,0,0])
        self.assertGreater(res.results.results[0].confidence, 0.5)

        res.run_classify_exhaust()
        self.assertEqual(len(res.results.results),1)
        self.assertEqual(res.results.results[0].name, 'sports')
        self.assertNotEquals(res.results.results[0].location, [0,0,0,0])
        self.assertGreater(res.results.results[0].confidence, 0.5)

        res.set_url(valid_link, 0.999)
        res.run_classify_exhaust()
        self.assertEqual(res.results, "unknown")

    def testMultipleQuery(self):
        filename = './imagelist.txt'
        q = myres.myquery(filename)
        q.set_threshold(0.2)
        search_res = myres.result_list()
        search_res.run_queries(q)
        self.assertEqual(len(search_res.results),q.count)
        self.assertEqual(search_res.results[0].results.results[0].name, 'lesser')
        self.assertGreater(search_res.results[0].results.results[0].confidence, 0.2)

        self.assertEqual(search_res.results[1].results, 'cannot download image.')
        self.assertEqual(search_res.results[2].results, 'unknown')
        self.assertEqual(search_res.results[3].results, 'invalid URL')
        self.assertEqual(search_res.results[4].results, 'cannot download image.')

        self.assertEqual(len(search_res.results[5].results.results), 2)
        self.assertEqual(search_res.results[5].results.results[0].name, 'Siamese')
        self.assertGreater(search_res.results[5].results.results[0].confidence, 0.2)
        self.assertEqual(search_res.results[5].results.results[1].name, 'wombat')
        self.assertGreater(search_res.results[5].results.results[1].confidence, 0.2)
        self.assertGreater(search_res.results[5].results.results[0].confidence, search_res.results[5].results.results[1].confidence)
if __name__ == '__main__':
    unittest.main()
    x = 0
    print('done')