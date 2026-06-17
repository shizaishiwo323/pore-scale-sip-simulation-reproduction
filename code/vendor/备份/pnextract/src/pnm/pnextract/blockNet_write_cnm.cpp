
#include "blockNet.h"


inline double randomG () /// to randomly distribute the shape factors, in case of errors
{
	double x1, x2, w, y;
	do{
		x1 = 2. * rand()/RAND_MAX - 1.;
		x2 = 2. * rand()/RAND_MAX - 1.;
		w = x1 * x1 + x2 * x2;
	} while (w >= 1.);
	w = sqrt ( (-2. * log(w)) / w);
	y = 0.00625 * (x1 * w + 5.);
	if (y > 0.049)	y = 0.0625;
	return y;
}


void  blockNetwork::writePNM() const
{
	/// pnflow uses the following indexes: [0:nB(=2)] for boundary nodes, 
	/// [nB:nP+nB] for internal nodes and throat indices start afterwards.
	/// here, in Statoil format, all these are subtracted by 1 (starting from -1).


#define sync_mne_mnf
#ifdef sync_mne_mnf

	///### First we compute the classical network model parameters,
	///#### throat radii, shape factors and lengths,
	vector<double> t_radiuss(nTrots,0.);//
	vector<double> t_shapeFacts(nTrots,0.);//
	vector<double> t_lengthP1toP2s(nTrots,0.);
	vector<double> t_lp1s(nTrots,0.);//
	vector<double> t_lp2s(nTrots,0.);//
	vector<double> t_ltrot(nTrots,0.); // throat portion of t_lengthP1toP2s

	///#### pore radii, shape factors and lengths.
	vector<double> p_radiuss(nNodes,0.);
	vector<double> p_shape1s(nNodes,0.);
	vector<double> p_physlengths(nNodes,0.);

	dbl3 checkSumAt(0.,0.,0.);
	cout<<"\ncalcThroats:"<<endl;
	int lengthP1toP2Warnings = 0;
	double nBelowAllowedG(0.), nAboveAllowedG(0.), totalArea(0.);


	///### Compute throat  parameters
	for (int ti=0; ti<nTrots; ++ti)  {
	    throatNE& tr = *throatIs[ti];

		double lthroat = 0;
		double lp1 = 0;
		double lp2 = 0;


		if (tr.surfaceArea == 0)		tr.surfaceArea = 6;
		if (mag(tr.CrosArea) <0.01)		tr.CrosArea[0] = 0.1;
		checkSumAt+=tr.CrosArea;

		/// - compute distance between the throat centre and each of the two adjacent pore centres
		double lpt1 = (   (tr.e1 <2)  ?  (tr.e1 == 0 ? tr.mb22()->fi:cg.nx-tr.mb22()->fi) : dist (poreIs[tr.e1]->mb, tr.mb22())   );
		double rp1 = std::max(   (tr.e1 <2 ) ? tr.mb22()->R : poreIs[tr.e1]->mb->R , 1.0f );
		double lpt2 = (   (tr.e2 <2)  ?  (tr.e2 == 0?tr.mb22()->fi:cg.nx-tr.mb22()->fi) : dist (poreIs[tr.e2]->mb, tr.mb22())   );
		double rp2 = std::max(   (tr.e2 <2 ) ? tr.mb22()->R : poreIs[tr.e2]->mb->R , 1.0f );

		/// - throat radius is the radius of the largest maximal sphere  on the throat surface
		double rr = std::max(tr.mb22()->R,0.5f);
		rr = std::min(std::min(rr,rp1),rp2);
		t_radiuss[ti] = rr+0.5*(0.5-double(rand())/RAND_MAX);
		/// - throat total length is the sum of the two half-throat lengths
		double lengthP1toP2 = lpt1+lpt2;
		if (lengthP1toP2 < 3.) 	lengthP1toP2 = 3.01, ++lengthP1toP2Warnings;

		/// - each pore is given 67% of the total throat length, the rest is called the throat elngth
		lp1 = lpt1*0.67;
		lp2 = lpt2*0.67;
		if (tr.e1 < 2 )	lp1 = 1;
		if (tr.e2 < 2 )	lp2 = 1;
		lthroat = lengthP1toP2-lp1-lp2;


		if (lthroat < 0.0000001) 	lthroat = 1;

		t_shapeFacts[ti] = rr*rr/4./mag(tr.CrosArea);  ///- new throat shape factor definition G = R^2/4A


		if (t_shapeFacts[ti]>= 0.09 )  {t_shapeFacts[ti] = std::min(0.079,t_shapeFacts[ti]/2.); nAboveAllowedG += mag(tr.CrosArea);}//. shape factor can not be this large, error: probably shared throat, temporary fix to handle in the flow code

		if (t_shapeFacts[ti]<0.01)//. shape factor can not be this small,
			{t_shapeFacts[ti] = std::max(randomG(),0.01); nBelowAllowedG += mag(tr.CrosArea);}

		totalArea += mag(tr.CrosArea);

		t_lengthP1toP2s[ti] = lengthP1toP2*1.;
		t_lp1s[ti] = lp1*1.;
		t_lp2s[ti] = lp2*1.;
		t_ltrot[ti] = lthroat*1;
	}
	cout<<  " P1-to-P2 length < 3    for " <<lengthP1toP2Warnings<<" throats"<<endl;
	cout<<" shapefactor: belowAllowedG "<<nBelowAllowedG/totalArea*100<<"%   aboveAllowedG "<<nAboveAllowedG/totalArea*100<<"%"<<endl;
	cout<<" checkSumAt: "<<checkSumAt<<endl;


	 cout<<"calc Pores"<<endl;


	///### Compute pore  parameters
	for (int pid=2; pid<nNodes; ++pid)  {
		poreNE& por = *poreIs[pid];
		double radius = por.mb->R;
		p_radiuss[pid] = radius;
		if (por.surfaceArea<1) por.surfaceArea = 6;
		if (por.volumn<1) por.volumn = 1;

		/// - pore shape factor is computed from a weighted average of its throat shape factors
		double shapeFactor(5.e-38), SumTArea(1e-36);
		for (const auto& bi:por.contacts)  {
			throatNE& tr = *throatIs[bi.second];
			shapeFactor += t_shapeFacts[bi.second]*mag(tr.CrosArea);
			SumTArea += mag(tr.CrosArea);
		}
		shapeFactor /= SumTArea;

		double porA(radius*radius/4./shapeFactor);
		SumTArea += porA;
		double pVol = por.volumn;
		por.volumn = pVol * porA/SumTArea;

		for (const auto& bi:por.contacts)  {
			throatNE& tr = *throatIs[bi.second];
			tr.volumn += pVol * mag(tr.CrosArea)/SumTArea;
		}

		p_shape1s[pid] = shapeFactor;//(por.volumn*poreLengthMax*2/por.surfaceArea/por.surfaceArea);
	}
#endif //sync_mne_mnf



	if (nBP6!=2) { cout<<"Skipping write of incompatible old network format."<<endl; return; }


	const double dx = cg.vxlSize;
	cout<<"Writing throats";cout.flush();


	struct ContactSplitRecord {
		int originalTi;
		int e1;
		int e2;
		double radius;
		double shapeFactor;
		double lengthP1toP2;
		double lp1;
		double lp2;
		double lthroat;
		double volume;
		int patchIndex;
		int patchCount;
		int faceCount;
	};
	vector<ContactSplitRecord> contactSplitRecords;
	contactSplitRecords.reserve(throatIs.size()*2);

	for (int ti = 0; ti < int(throatIs.size()); ++ti) {
		const throatNE& tr = *throatIs[ti];
		vector<voxel*> patchVoxels;
		patchVoxels.reserve(tr.toxels1.size()+tr.toxels2.size());
		for (auto vi: tr.toxels1) if (vi) patchVoxels.push_back(vi);
		for (auto vi: tr.toxels2) if (vi) patchVoxels.push_back(vi);

		const bool canSplitInternal = (tr.e1 >= 2 && tr.e2 >= 2 && patchVoxels.size() > 1);
		vector<vector<int> > components;
		if (canSplitInternal) {
			vector<int> visited(patchVoxels.size(), 0);
			for (int start = 0; start < int(patchVoxels.size()); ++start) {
				if (visited[start]) continue;
				vector<int> stack;
				vector<int> component;
				stack.push_back(start);
				visited[start] = 1;
				while (!stack.empty()) {
					int current = stack.back();
					stack.pop_back();
					component.push_back(current);
					voxel* vc = patchVoxels[current];
					for (int other = 0; other < int(patchVoxels.size()); ++other) {
						if (visited[other]) continue;
						voxel* vo = patchVoxels[other];
						int di = std::abs(vc->i - vo->i);
						int dj = std::abs(vc->j - vo->j);
						int dk = std::abs(vc->k - vo->k);
						if (di <= 1 && dj <= 1 && dk <= 1 && (di + dj + dk) <= 2) {
							visited[other] = 1;
							stack.push_back(other);
						}
					}
				}
				components.push_back(component);
			}
		}
		if (components.empty()) {
			components.push_back(vector<int>());
			for (int i = 0; i < int(patchVoxels.size()); ++i) components.back().push_back(i);
		}

		const double minThroatLengthVoxels = 0.25;
		const int totalFaces = std::max(1, int(patchVoxels.size()));
		const double originalShapeFactor = std::max(t_shapeFacts[ti], 1.e-12);
		const double originalLength = std::max(t_ltrot[ti], minThroatLengthVoxels);
		const double originalArea = t_radiuss[ti]*t_radiuss[ti]/4./originalShapeFactor;
		const double originalAreaOverLength = originalArea/originalLength;
		const int patchCount = int(components.size());
		for (int ci = 0; ci < patchCount; ++ci) {
			const vector<int>& comp = components[ci];
			double centroidI = 0., centroidJ = 0., centroidK = 0.;
			double maxR = 0.5;
			for (int idx: comp) {
				voxel* vi = patchVoxels[idx];
				centroidI += vi->i + 0.5;
				centroidJ += vi->j + 0.5;
				centroidK += vi->k + 0.5;
				maxR = std::max(maxR, double(vi->R));
			}
			const int faceCount = std::max(1, int(comp.size()));
			centroidI /= faceCount;
			centroidJ /= faceCount;
			centroidK /= faceCount;

			double splitLength = t_ltrot[ti];
			double splitTotalLength = t_lengthP1toP2s[ti];
			double splitLp1 = t_lp1s[ti];
			double splitLp2 = t_lp2s[ti];
			double splitRadius = t_radiuss[ti];
			double splitShapeFactor = t_shapeFacts[ti];
			if (canSplitInternal) {
				const poreNE* p1 = poreIs[tr.e1];
				const poreNE* p2 = poreIs[tr.e2];
				double d1 = std::sqrt((centroidI-p1->mb->fi)*(centroidI-p1->mb->fi)
				                    +(centroidJ-p1->mb->fj)*(centroidJ-p1->mb->fj)
				                    +(centroidK-p1->mb->fk)*(centroidK-p1->mb->fk));
				double d2 = std::sqrt((centroidI-p2->mb->fi)*(centroidI-p2->mb->fi)
				                    +(centroidJ-p2->mb->fj)*(centroidJ-p2->mb->fj)
				                    +(centroidK-p2->mb->fk)*(centroidK-p2->mb->fk));
				double rp1 = std::max(double(p1->mb->R), 1.0);
				double rp2 = std::max(double(p2->mb->R), 1.0);
				splitLp1 = std::max(d1-rp1, 0.);
				splitLp2 = std::max(d2-rp2, 0.);
				splitLength = std::max(d1+d2-rp1-rp2, minThroatLengthVoxels);
				splitTotalLength = splitLp1 + splitLp2 + splitLength;
				const double contactWeight = double(faceCount)/double(totalFaces);
				const double targetAreaOverLength = originalAreaOverLength * contactWeight;
				const double targetArea = std::max(targetAreaOverLength * splitLength, 1.e-12);
				splitShapeFactor = std::min(0.079, std::max(0.01, originalShapeFactor));
				splitRadius = std::sqrt(4.0 * splitShapeFactor * targetArea);
				splitRadius = std::max(splitRadius, 0.5);
			}

			ContactSplitRecord rec;
			rec.originalTi = ti;
			rec.e1 = tr.e1;
			rec.e2 = tr.e2;
			rec.radius = splitRadius;
			rec.shapeFactor = splitShapeFactor;
			rec.lengthP1toP2 = splitTotalLength;
			rec.lp1 = splitLp1;
			rec.lp2 = splitLp2;
			rec.lthroat = splitLength;
			rec.volume = tr.volumn * double(faceCount) / double(totalFaces);
			rec.patchIndex = ci + 1;
			rec.patchCount = patchCount;
			rec.faceCount = faceCount;
			contactSplitRecords.push_back(rec);
		}
	}

	cout<<"Writing contact patch split throats";cout.flush();
	{ ///### write  _link1.dat file
		FILE* fil = fopen((cg.name() + "_link1.dat").c_str(), "w");
		fprintf(fil, "%6d\n", int(contactSplitRecords.size()));
		for (int si = 0; si < int(contactSplitRecords.size()); ++si) {
			const ContactSplitRecord& rec = contactSplitRecords[si];
			fprintf(fil, "%6d %6d %6d %E %E %E\n", si+1, int(rec.e1-1), int(rec.e2-1),
						  rec.radius*dx, rec.shapeFactor, rec.lengthP1toP2*dx);
		}
		fclose(fil);
	}

	{///### write  _link2.dat file
		FILE* fil = fopen((cg.name() + "_link2.dat").c_str(), "w");
		for (int si = 0; si < int(contactSplitRecords.size()); ++si) {
			const ContactSplitRecord& rec = contactSplitRecords[si];
			fprintf(fil, "%6d %6d %6d %E %E %E %E %E\n", si+1, int(rec.e1-1), int(rec.e2-1),
			  rec.lp1*dx, rec.lp2*dx, rec.lthroat*dx, rec.volume*dx*dx*dx, 0.);
		}
		fclose(fil);
	}
	cout<<" ("<<contactSplitRecords.size()<<" split records from "<<throatIs.size()<<" aggregated throats)."<<"\n";cout.flush();

	{
		FILE* fil = fopen((cg.name() + "_contact_patch_split_diagnostics.csv").c_str(), "w");
		fprintf(fil, "split_throat_id,original_throat_id,pore1_id,pore2_id,patch_index,patch_count,face_count,length_voxels,radius_voxels,shape_factor,volume_voxels3\n");
		for (int si = 0; si < int(contactSplitRecords.size()); ++si) {
			const ContactSplitRecord& rec = contactSplitRecords[si];
			fprintf(fil, "%d,%d,%d,%d,%d,%d,%d,%.12g,%.12g,%.12g,%.12g\n",
				si+1, rec.originalTi+1, int(rec.e1-1), int(rec.e2-1), rec.patchIndex, rec.patchCount,
				rec.faceCount, rec.lthroat, rec.radius, rec.shapeFactor, rec.volume);
		}
		fclose(fil);
	}




	cout<<"Writing pores";cout.flush();
	{///### write  _node1.dat file
		FILE* fil = fopen((cg.name() + "_node1.dat").c_str(), "w");
		fprintf(fil, "%6d  %E  %E  %E\n", int(poreIs.size())-2, cg.nx*dx, cg.ny*dx, cg.nz*dx);

		for (int pid = 2; pid < int(poreIs.size()); ++pid)//. 0th and first are inlet and outlet elements
		{	const poreNE& por = *poreIs[pid];

			// pnflow does not work with X0
			fprintf(fil, "%6d %E %E %E %3d", int(pid-1), por.mb->fi*dx+0.*cg.X0[0], por.mb->fj*dx+0.*cg.X0[1], por.mb->fk*dx+0.*cg.X0[2], int(por.contacts.size()));

			int inlet = 0,outlet = 0;
			for (std::map<int,int>::const_iterator bi = por.contacts.begin(); bi != por.contacts.end(); ++bi)  {
				const throatNE* tb = throatIs[bi->second];
				if (tb->e1 == pid)  {
					if (tb->e2 == 0)       inlet = 1;
					else if (tb->e2 == 1)  outlet = 1;

					fprintf(fil, "\t%5d", int(tb->e2-1));
				}
				else
				{
					if (tb->e1 == 0)		 inlet = 1;
					else if (tb->e1 == 1)  outlet = 1;
					fprintf(fil, "\t%5d", int(tb->e1-1));
				}
			}

			fprintf(fil, "\t%5d", (inlet) ); //. .dummy
			fprintf(fil, "\t%5d", (outlet) ); //. .dummy

			for (std::map<int,int>::const_iterator bi = por.contacts.begin(); bi != por.contacts.end(); ++bi)  {
				fprintf(fil, "\t%5d", int(bi->second)+1);
			}
			fputs("\n", fil);
		}
		fclose(fil);
	}

	{///### write  _node2.dat file
		FILE* fil = fopen((cg.name() + "_node2.dat").c_str(), "w");
		for (int pid = 2; pid < int(poreIs.size()); ++pid)//. 0th and first are inlet and outlet elements
		{	const poreNE& por = *poreIs[pid];
			fprintf(fil, "%6d %E %E %E %E\n", int(pid-1),  por.volumn*dx*dx*dx,  por.radius()*dx, p_shape1s[pid], 0.);
		}
		fclose(fil);
	}

	cout<<".\n";cout.flush();

}






