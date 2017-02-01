 
 
c---------------------------------------------------------
c
c img2xym_HRC.e
c
c
c This routine will find stars in an HRC image and will
c measure them with the specified PSF.  To execute, type
c the name of the routine without arguments, then it
c will tell you the arguments that need to be specified.
c
c Feel free to e-mail jay@eeyore.rice.edu if you have
c any questions.
c
c It is designed to work on either the *_flt.fits images
c or images I call *_H.fits, which are the flt images turned
c into integer*4 images, so that they can be compressed
c more efficiently.  This routine even works on the *_drz.fits
c images, though psf-fitting to _drz images is not recommended...
c one should just do aperture photometry on these images (which
c can help with the calibration).
c
c The output of the program is the raw, chip based x,y
c position and instrumental magnitude (m=-2.5*log10[flux/DN])
c for each peak in the image that meets the finding criteria.
c I also report the distortion-corrected position and the
c pixel-area corrected instrumental magnitude.
c
c October 2007
c * modified the fits reader to work for MAC OSX
c   (should still work under LINUX/SOLARIS/ETC)
c * also added the QSEL parameter, which allows you
c   to only include objects that look at least halfway
c   like a star
c
c---------------------------------------------------------
 
 
 
 
#define _NNLMAX_  1000000
#define _NMAXFIT_    9999
#define _NITMAX_      12
#define _NPMAX_        1
#define _NREGS_        9
 
#define _LOFLAG_    -00050
#define _HIFLAG_     30000
 
#define _PXDIMX_    1014
#define _PXDIMY_    1014
 
#define _LOGRES_    .false.
#define _LINUX_     .true.
 
 
      program img2xym
      implicit none
 
      character*200 STRING
      character*80  INFILE
      character*80  PSFFILE
      character*80  OUTPUT
      character*80  OUTPUTQ
      character*80  SATPUT
      character*80  DIRECT
      character*80  PREFIX
      character*7   SUFFIX
 
      character*80  QQQQ
 
      integer  ii,  jj
      integer   i,   j
      integer  iiu, jju
      integer iu
      integer ir
      real    rsum(10), rr
      integer hobs, fnd_hloc
      integer HMIN
      real    PMAX
      real    FMIN
 
      integer iargc, NARG, NARGs
      integer NOUT
 
      character*70 INFO(10)
      common / fitsinfo / INFO
      integer IFILT
 
      real*8 x, y, z, s
      real*8 mz
      real*8 xx,yy,mzz
      real   hrc_zpa
      real   apphot
 
      real*8 dclip
      real hmb, histmode
 
      real qofminpct
      common/qofmin/qofminpct
 
      real      pix(_PXDIMX_,_PXDIMY_)
      integer*4 pox(_PXDIMX_,_PXDIMY_)
 
      real locsky
      real flxest
 
      logical MASK
 
      real psf(101,101)
 
      real*8  sum
 
      common /narginfo/narg
 
      logical DOAPPHOT
      data    DOAPPHOT / .false. /
      real    RAP
      integer SKI,SKO
 
      logical DOSUB
      logical DOQSEL
      real rpsf_phot, r, dx, dy
 
      logical ISDRIZZLE
 
      integer ipsf, jpsf
      real    xpsf, ypsf
      real*8  psum
      real    wfc3ir_zpa
 
      if (iargc().lt.5) then
         print*,'                                     '
         print*,'This routine takes 5 args:           '
         print*,'                                     '
         print*,
     .   'img2xym_wfc3ir HMIN FMIN PMAX PSFFILE [list*]_H.fits'
         print*,'                                     '
         print*,'     HMIN: dominance of peak to be found; '
         print*,'           the closest brighter pixel than this'
         print*,'           one must more than HMIN pixels away'
         print*,'                                     '
         print*,'     FMIN: min peak brightness over  '
         print*,'           sky (total in 3x3 box)    '
         print*,'                                     '
         print*,'     PMAX: max pixel included (if .lt.1e9'
         print*,'           then do not measure satd stars)'
         print*,'                                     '
         print*,'  PSFFILE: to use to measure         '
         print*,'           (a fits image; eg PSFHRC.F475W.fits)'
         print*,'       -or-                          '
         print*,'           you can specify an aperture to use'
         print*,'           for aperture photometry:          '
         print*,'           "APPHOT 2.99 6 9"                 '
         print*,'               uses an ap of r<2.99 and      '
         print*,'               takes sky from between 6 and 9'
         print*,'                                             '
         print*,'   *.fits: sequence of .fits file HRC images'
         print*,'           can be either *_flt.fits or *_H.fits '
         print*,'           (the latter format is simply a '
         print*,'           1014x1014 integer*4 image, which I '
         print*,'           find compresses better than _flt images'
         print*,'                                     '
         print*,'                                     '
         print*,'   COMPILED PARAMETERS:              '
         print*,'                                     '
         print*,'      LOFLAG: ',_LOFLAG_
         print*,'      HIFLAG: ',_HIFLAG_
         print*,'      PXDIMX: ',_PXDIMX_
         print*,'      PXDIMY: ',_PXDIMY_
         print*,' '
         print*,' '
         print*,'   GOOD VALUES   HMIN =      5 '
         print*,'   TO USE:       FMIN =    100 '
         print*,'                 PMAX =   1e15 '
         print*,' '
         print*,' '
         stop
         endif
 
      call getarg(1,STRING)
      read(STRING,*) HMIN
 
      call getarg(2,STRING)
      read(STRING,*) FMIN
 
      call getarg(3,STRING)
      read(STRING,*) PMAX
      MASK = .true.
      if (PMAX.lt.0) then
         MASK = .false.
         PMAX = abs(PMAX)
         endif
 
      call getarg(4,PSFFILE)
 
      if (PSFFILE(1:6).eq.'APPHOT') then
         write(*,'(''PSFFILE : '',80a)') PSFFILE
         read(PSFFILE(7:80),*) RAP,SKI,SKO
         write(*,'('' ---> APERTURE RAP: '',f8.4)') RAP
         write(*,'('' --->    INNER SKI: '',i3)') SKI
         write(*,'('' --->    OUTER SKO: '',i3)') SKO
         DOAPPHOT = .true.
        else
         print*,'OPEN: ',PSFFILE(1:15)
         call readfits_r4(PSFFILE,psf,101,101)
        endif
 
 
      print*,' ',HMIN
      print*,' ',FMIN
      print*,' ',PMAX
 
 
      NARGs = iargc()
 
      DOSUB = .false.
      do NARG = 5, NARGs
 
         call getarg(NARG,infile)
         if (infile(1:4).eq.'SUBT') then
            DOSUB = .true.
            goto 9
            endif
         if (infile(1:4).eq.'QSEL') then
            DOQSEL = .true.
            goto 9
            endif
         call dirstrip(infile,direct,prefix,suffix)
 
 
         ISDRIZZLE = .false.
 
c        call readfits_i4r(infile,pix,1014,1014)
         do i = 1, 1014
         do j = 1, 1014
            pix(i,j) = 0.00
            enddo
            enddo
 
         call read_wfc3ir_flt(infile,pix)
 
         if (DOSUB) then
            do i = 1, 1014
            do j = 1, 1014
               pox(i,j) = pix(i,j)+0.5
               enddo
               enddo
            endif
 
         IFILT = 0
 
         write(*,'(''  FILTER: '',40a)') INFO(2)
         write(*,'(''   IFILT: '',i2 )') IFILT
 
         iu = -1
         do i = 1,76
            if (prefix(i:i).ne.' ') iu = i
            enddo
         output  = prefix(1:iu) // '.xym'
         outputq = prefix(1:iu) // '.xymq'
 
         write(*,'(79(''-''))')
 
         write(*,'(''  prefix: '',40a)') prefix
         write(*,'('' outfile: '',40a)') output
 
 
         print*,'HISTMODE...'
         hmb = histmode(100,450,100,450,pix)
 
         write(*,'(a20,2x,f8.1,2x,a20)') infile,hmb,output
 
         open(77,file=output ,status='unknown')
c        open(78,file=outputq,status='unknown')
         write(77,'(''#-------------------------------------'')')
         write(77,'(''#     '')')
         write(77,'(''# THIS FILE:  '',80a)') output
         write(77,'(''#     '')')
         write(77,'(''#      HMIN:  '',i7)') HMIN
         write(77,'(''#      PMAX:  '',e12.6)') PMAX
         write(77,'(''#      FMIN:  '',f9.1)') FMIN
         write(77,'(''#       PSF:  '',80a)') PSFFILE
         write(77,'(''#       IMG:  '',80a)') INFILE
         write(77,'(''#    SKYBAR:  '',f9.1)') hmb
         write(77,'(''#    FILTER:  '',80a)') INFO(2)
         write(77,'(''#     IFILT:  '',i7)') IFILT
         write(77,'(''#     '')')
         write(77,'(''#-------------------------------------'')')
         write(77,'(''#     '')')
         write(77,'(''#........ ......... ........   '',
     .              ''......... ......... ........ '')')
         write(77,'(''#   xraw      yraw     mraw    ''
     .              ''    xcor      ycor     mcor   qsel '')')
         write(77,'(''#........ ......... ........   '',
     .              ''......... ......... ........ '')')
 
         write(77,'(''# '')')
         write(77,'(''# PSF ENCLOSED ENERGY... '')')
         write(77,'(''# '')')
         do ir = 1, 12
            psum = 0.
            do ipsf = 001, 101
            do jpsf = 001, 101
               xpsf = 0.25*(ipsf-51)
               ypsf = 0.25*(jpsf-51)
               if (xpsf**2+ypsf**2.lt.ir**2) then
                  psum = psum + psf(ipsf,jpsf)
                  endif
               enddo
               enddo
            write(77,'(''# '',i2.2,1x,f10.7)') ir,psum/16.0
            enddo
         write(77,'(''# '')')
         write(77,'(''# '')')
 
 
         print*,'GO THRU IMAGE ROW BY ROW...'
         print*,'      PMAX: ',PMAX
         print*,'      FMIN: ',FMIN
         print*,'    PSFIMG: ',psf(051,051)
         print*,'    PIXIMG: ',pix(512,512)
         print*,'    SKYBAR: ',hmb
         NOUT = 0
         do jju = 10, _PXDIMY_-09
         do iiu = 10, _PXDIMX_-09
c        do jju = 509, 509
c        do iiu = 504, 504
            ii = iiu
            jj = jju
            if (pix(ii,jj).lt._LOFLAG_+0.5) goto 444
            if (pix(ii,jj).gt.PMAX) goto 444
            do i = ii-1,ii+1
            do j = jj-1,jj+1
               if (pix(i,j).gt.pix(ii,jj)) goto 444
               enddo
               enddo
            locsky = pix(ii,jj)
            do i = ii-2,ii+2
            do j = jj-2,jj+2
               if (pix(i,j).le._LOFLAG_+0.5) goto 444
               if (pix(i,j).lt.locsky) locsky = pix(i,j)
               enddo
               enddo
            if (pix(ii,jj).gt._HIFLAG_) locsky = hmb
            flxest = pix(ii,jj) - 4*locsky +
     .               max(pix(ii+1,jj)+pix(ii+1,jj+1)+pix(ii,jj+1),
     .                   pix(ii-1,jj)+pix(ii-1,jj+1)+pix(ii,jj+1),
     .                   pix(ii+1,jj)+pix(ii+1,jj-1)+pix(ii,jj-1),
     .                   pix(ii-1,jj)+pix(ii-1,jj-1)+pix(ii,jj-1))
            if (flxest.lt.FMIN) goto 444
            hobs = fnd_hloc(ii,jj,pix)
            if (hobs.lt.HMIN) goto 444
 
            x = ii
            y = jj
            z = flxest
 
            if (DOAPPHOT) then
	          x = ii + (pix(ii+1,jj)-pix(ii-1,jj))/2./
     .                     (pix(ii,jj)-min(pix(ii+1,jj),pix(ii-1,jj)))
	          y = jj + (pix(ii,jj+1)-pix(ii,jj-1))/2./
     .                     (pix(ii,jj)-min(pix(ii,jj+1),pix(ii,jj-1)))
                  locsky = hmb
                  z = apphot(ii,jj,RAP,SKI,SKO,pix,locsky)
                  s = locsky
               else
                 call find_xyz(x,y,z,s,pix,psf)
                 if (DOQSEL.and.qofminpct.lt.0.00) goto 444
                 if (DOQSEL.and.qofminpct.gt.0.50) goto 444
               endif
 
 
            if (ii.gt.12.and.ii.lt.1014-12.and.
     .          jj.gt.12.and.jj.lt.1014-12) then
                do ir = 01, 10
                   sum = 0.
                   do i = -11,+11
                   do j = -11,+11
                      rr = sqrt((i-0.)**2+(j-0.)**2)
                      if (rr.lt.ir) sum = sum+pix(ii+i,jj+j)-s
                      enddo
                      enddo
                   rsum(ir) = sum
                   enddo!ir
                endif
 
            z = dclip(z,1.0d00,1.0d20)
            mz = -2.5*log10(z)
            NOUT = NOUT + 1
            if (NOUT.eq.NOUT/100*100.or.pix(ii,jj).gt._HIFLAG_) then
                write( *,178) NOUT,x,y,mz,s,
     .                       pix(ii  ,jj  ),
     .                       pix(ii+1,jj  ),pix(ii-1,jj  ),
     .                       pix(ii  ,jj+1),pix(ii  ,jj-1)
                endif
            xx  = x
            yy  = y
            mzz = mz + wfc3ir_zpa(xx,yy)
            call wfc3ir_gcX(x,y,xx,yy)
            write(77,177) x,y,mz,xx,yy,mzz,qofminpct
 177        format(f9.3,1x,f9.3,1x,f8.3,3x,
     .             f9.3,1x,f9.3,1x,f8.3,1x,f10.4)
 178        format(2x,i5,1x,f9.3,1x,f9.3,1x,
     .             f8.3,1x,f8.3,'|',f15.0,'|',4f15.0,'|',4f15.0)
            if (DOSUB) then
               do i = max(0001,ii-15),min(1014,ii+15)
               do j = max(0001,jj-15),min(1014,jj+15)
                  dx = i-x
                  dy = j-y
                  r = rpsf_phot(dx,dy,psf)
                  if (pix(i,j).lt.119000) pox(i,j) = pox(i,j) - r*z
                  enddo
                  enddo
               endif
 
 
            pix(ii,jj) = pix(ii,jj) + 1 ! so that we don't find 2 stars in the same place
 444        continue                    ! if there are two equal-valued pixels next to each
            enddo                       ! other
            enddo
         close(77)
c        close(78)
         if (DOSUB) then
            QQQQ = 'sub.' // INFILE
            call writfits_i4(QQQQ,pox,1014,1014)
            endif
   9     continue
         enddo!NARG
 
 
       end
 
 
c----------------------------------------------------
c
c SUBROUTINES
c
c----------------------------------------------------
 
 
c----------------------------------------------------
c
c read in a file name and strip off the directory info
c
      subroutine dirstrip(FULL,DIRECT,PREFIX,SUFFIX)
      implicit none
 
 
      character*80 FULL
      character*80 DIRECT
      character*80 PREFIX
      character*7  SUFFIX
 
      integer i
 
      integer ndir
      integer ndot
      integer nbrk
      integer nend
 
      ndir = 0
      ndot = 0
      nbrk = 0
      nend = 0
      do i = 1,80
         if (FULL(i:i).eq.'/')  ndir = i
         if (FULL(i:i).eq.'.')  ndot = i
         if (FULL(i:i).eq.'[')  nbrk = i
         if (FULL(i:i).eq.' '.and.
     .       nend.eq.0)  nend = i-1
         enddo
 
      if (ndot.eq.0) then
         print*,'THE IMAGE MUST HAVE A DOT IN ITS NAME.'
         print*,'NAME: ',FULL
         stop
         endif
 
 
      DIRECT = FULL(1:ndir)
      PREFIX = FULL(ndir+1:ndot-1)
      SUFFIX = FULL(ndot+1:nend  )
 
      write(*,'(''DRCTRY:'',i3,3x,80a)')
     .            ndir       ,DIRECT(1:ndir)
      write(*,'(''PREFIX:'',i3,3x,80a)')
     .            ndot-ndir-1,PREFIX(1:ndot-ndir-1)
      write(*,'(''SUFFIX:'',i3,3x,80a)')
     .            nend-ndot  ,SUFFIX(1:nend-ndot)
 
      return
      end
 
 
 
c-------------------------------------------------
c
c find how far out you have to go from one pixel
c to find a brighter one
c
      integer function fnd_hloc(i,j,pix)
      implicit none
 
      integer i, j
      real pix(_PXDIMX_,_PXDIMY_)
 
      integer h
      integer ii, jj
      logical inimage
 
      do h = 1, 8
         do ii = i-h,i+h
         do jj = j-h,j+h
            if (.not.inimage(ii,jj)) goto 1
            if (pix(ii,jj).gt.pix(i,j)) goto 1
            enddo!jj
            enddo!ii
         enddo!h
 
 1    continue
      fnd_hloc = h
      return
      end
 
 
c------------------------------------------------------
c
c bubble-sorts real numbers in ascending order
c
      subroutine rbubble(r1,NTOT)
      implicit none
      real r1(1)
      real temp
      integer NTOT
      integer n
      logical change
 777  continue
      change = .false.
      do n = 1, NTOT-1
         if (r1(n).gt.r1(n+1)) then
            temp = r1(n)
            r1(n) = r1(n+1)
            r1(n+1) = temp
            change = .true.
            endif
         enddo
      if (change) goto 777
      end
 
c------------------------------------------------------
c
c bubble-sorts real*8 numbers in ascending order
c
      subroutine dbubble(r1,NTOT)
      implicit none
      real*8 r1(1)
      integer NTOT
      integer n
      real*8 temp
      logical change
 777  continue
      change = .false.
      do n = 1, NTOT-1
         if (r1(n).gt.r1(n+1)) then
            temp = r1(n)
            r1(n) = r1(n+1)
            r1(n+1) = temp
            change = .true.
            endif
         enddo
      if (change) goto 777
      end
 
 
 
c------------------------------------------------------
c
c uses a histogram method to find the mode in the
c region (i1:i2,j1:j2)
c
      real function histmode(i1,i2,j1,j2,pixarr)
      implicit none
 
      integer i1,i2,j1,j2
      real pixarr(_PXDIMX_,_PXDIMY_)
 
      integer i, j, h
      integer mhi, mlo
      real hist_summ
 
      integer hist(5000)
      integer hcum(5000)
      integer NTOT
      real root
 
      real    pcum(101)
      integer ipc
      integer imin
      real    dmin
      real    ptot
 
      do h = 1,5000
         hist(h) = 0
         hcum(h) = 0
         enddo
 
      NTOT = 0
      PTOT = 0
      do i = i1,i2
      do j = j1,j2
         if (pixarr(i,j).ge._HIFLAG_) goto 2
         if (pixarr(i,j).le._LOFLAG_) goto 2
         h = pixarr(i,j)+10.5
         if (h.lt.   1) goto 2
         if (h.gt.5000) goto 2
         hist(h) = hist(h) + 1
         NTOT = NTOT + 1
         PTOT = PTOT + pixarr(i,j)
 2       continue
         enddo
         enddo
 
 
      hcum(1) = hist(1)/2
      do h = 2, 5000
         hcum(h) = hcum(h-1) + (hist(h)+hist(h-1))/2
         ipc = 100.*hcum(h)/NTOT + 1.5
         do i = ipc, 101
            pcum(i) = h
            enddo
         enddo
      do ipc = 1, 101
         pcum(ipc) = root(NTOT*(ipc-1)/100.,hcum,5000)
         enddo
 
      dmin = pcum(40+1)-pcum(1)
      imin = 1
      do i = 2, 101-40
         if (pcum(40+i)-pcum(i).lt.dmin) then
            dmin = pcum(40+i)-pcum(i)
            imin = i
            endif
         enddo
 
      mhi = pcum(40+imin)+3
      mlo = pcum(imin)   -3
 
      histmode = 0.
      hist_summ = 0.
      do h = mlo, mhi
         hist_summ = hist_summ + hist(h)
         histmode = histmode + hist(h)*(h-10)
 4       format(1x,i4,1x,i4,1x,i8,1x,f8.4,' %  ',
     .          1x,i8,1x,f8.4,' %  ')
        !print*,h,hist(h)
         enddo
      do h = mhi+1,mhi+(mhi-mlo)
         enddo
 
      histmode = histmode / hist_summ
 
      if (hist_summ.le.0) histmode = PTOT / NTOT
 
      NTOT = 0
      PTOT = 0
      do i = i1,i2
      do j = j1,j2
         if (pixarr(i,j).ge.(mlo-10).and.
     .       pixarr(i,j).le.(mhi-10)) then
             NTOT = NTOT + 1
             PTOT = PTOT + pixarr(i,j)
             endif
         enddo
         enddo
      histmode = PTOT/NTOT
 
      return
 
      end
 
 
c-----------------------------------------
c
c find where a function goes through zero
c
      real function root(n,list,NTOT)
      implicit none
      real    n
      integer list(1)
      integer NTOT
      integer nl
 
      do nl = 1, NTOT-2
         if (n.lt.list(nl+1)) goto 3
         enddo
 3    continue
 
      root = nl
      if (list(nl).ne.list(nl+1))
     .    root = nl + 1.*(n-list(nl))/(list(nl+1)-list(nl))
 
      return
      end
 
 
      real function mode_sky(ixc,iyc,inr,ior,pixarr)
      implicit none
 
      integer ixc, iyc
      integer inr, ior
      real pixarr(_PXDIMX_,_PXDIMY_)
 
      integer i  , j
      integer ixh, iyh
      real    rij
 
      logical inimage
      real sklist(9999)
      integer nsk
      real modelist
 
      nsk = 0
      do i = -ior+1, ior-1
      do j = -ior+1, ior-1
         rij = i**2+j**2
         if (rij.ge.ior**2) goto 333
         if (rij.lt.inr**2) goto 333
         ixh = ixc + i
         iyh = iyc + j
         if (.not.inimage(ixh,iyh)) goto 333
         if (pixarr(ixh,iyh).le._LOFLAG_) goto 333
         if (pixarr(ixh,iyh).ge._HIFLAG_) goto 333
         nsk = nsk + 1
         if (nsk.gt.9999) goto 333
         sklist(nsk) = pixarr(ixh,iyh)
 333     continue
         enddo
         enddo
 
      mode_sky = modelist(sklist,nsk,0.35)
 
 
      return
      end
 
 
 
 
 
 
 
 
c----------------------------------------------------------------
c
c find the mode of a list of numbers
c find the window that contains PCT
c find the modpoint of this window
c average all points within 1.5 windows of this
c point
c
      real function modelist(list,N,pct)
      implicit none
 
      integer N
      real list(N)
      real pct
 
      real*8 hold(20000)
      integer i, ip
      integer iwin
      real    dwin
      real    mwin, mmin, mmax
      real    htot
      integer nuse
 
 
      if (N.gt.20000) then
         print*,'modelist: ',N
         print*,'need to increase the array space'
         stop
         endif
 
      do i = 1, N
         hold(i) = list(i)
         enddo
 
      call dbubble(hold,N)
 
 
      ip = pct*N + 0.75
 
      iwin = 1
      dwin = hold(1+ip)-hold(1)
      do i = 2, N-ip
         if (hold(i+ip)-hold(i).lt.dwin) then
            iwin = i
            dwin = hold(i+ip)-hold(i)
            endif
         enddo
      mwin = 0.5*(hold(iwin)+hold(iwin+ip))
      mmin = mwin - 0.75*dwin
      mmax = mwin + 0.75*dwin
 
      nuse = 0.
      htot = 0
      do i = 1, N
         if (hold(i).ge.mmin.and.hold(i).le.mmax) then
            nuse = nuse + 1
            htot = htot + hold(i)
            endif
         enddo
      modelist = htot / nuse
 
      return
      end
 
 
c-------------------------------------------------------
c
c only analyze pixels within the array
c
      logical function inimage(i,j)
      implicit none
 
      integer i, j
 
      inimage = .true.
 
      if (i.lt.       1) inimage = .false.
      if (i.gt._PXDIMX_) inimage = .false.
      if (j.lt.       1) inimage = .false.
      if (j.gt._PXDIMY_) inimage = .false.
 
      return
      end
 
 
c--------------------------------------------
c
c make sure we don't get any out of bounds values
c
      real function rclip(rval,rlo,rhi)
      implicit none
 
      real rval
      real rlo
      real rhi
 
      rclip = rval
      if (rclip.gt.rhi) rclip = rhi
      if (rclip.lt.rlo) rclip = rlo
      if (.not.(rclip.lt.rhi).and.
     .    .not.(rclip.gt.rlo)) rclip = rhi
 
      return
      end
 
 
      real*8 function dclip(dval,dlo,dhi)
      implicit none
 
      real*8 dval
      real*8 dlo
      real*8 dhi
 
      dclip = dval
      if (dclip.gt.dhi) dclip = dhi
      if (dclip.lt.dlo) dclip = dlo
      if (.not.(dclip.lt.dhi).and.
     .    .not.(dclip.gt.dlo)) dclip = dhi
 
      return
      end
 
 
 
 
 
c******************************************************
c
c this routine will use the inner 5x5 pixels of a star
c to fit for the flux and position (x,y,z); it will
c determine the best sky value...
c
      subroutine find_xyz(x,y,z,s,pix,psf)
      implicit none
 
      real*8 x,y,z,s
      real pix(_PXDIMX_,_PXDIMY_)
      real psf(101,101)
 
      real pixarr(5,5)
      real psfarr(5,5)
      real wgtarr(5,5)
 
      real pixl(999)
      real psfl(999)
      integer m, oi, oj
 
      real wgtsum, qofsum
 
      real qofmin
      real qofminpct
      common/qofmin/qofminpct
      real qofdx, qofdy
      real qofflxp
      real qofflxw
      real qofflxa
      real dz
      real flx
 
      integer ix, iy
 
      integer i, j, ii, jj
      real dx0, dy0
      real dx,  dy
      integer NIT
      real scl
      real ddx, ddy, ddd
      real rpsf_phot, r
      real   zsig
 
      real*8 sumpf, sumff
      real*8 sumf , sump
      real   mode_sky
      real   omax
      real*8 sumx, sumy
 
      integer narg
      common /narginfo/narg
 
      logical first
      data first/.true./
 
      ix = int(x+0.5)
      iy = int(y+0.5)
 
 
      if (pix(ix,iy).gt._HIFLAG_) then
         s = mode_sky(ix,iy,15,25,pix)
         sumx = 0.
         sumy = 0.
         sumf = 0.
         do i = ix-2, ix+2
         do j = iy-2, iy+2
            sumx = sumx + i*(pix(i,j)-s)
            sumy = sumy + j*(pix(i,j)-s)
            sumf = sumf +   (pix(i,j)-s)
            enddo
            enddo
         x = sumx/sumf
         y = sumy/sumf
         z = sumf
         call find_xyz_SAT(x,y,z,s,pix,psf)
         return
         endif
 
      if (pix(ix,iy).le.    2000) then
           s = mode_sky(ix,iy,6,9,pix)
         else
           s = mode_sky(ix,iy,8,12,pix)
         endif
 
      omax = max(pix(ix+2,iy+0),pix(ix+2,iy+1),pix(ix+2,iy-1),
     .           pix(ix-2,iy+0),pix(ix-2,iy+1),pix(ix-2,iy-1),
     .           pix(ix-2,iy-2),pix(ix-2,iy+2),pix(ix+2,iy+2),
     .                                         pix(ix+2,iy-2),
     .           pix(ix+0,iy+2),pix(ix+1,iy+2),pix(ix-1,iy+2),
     .           pix(ix+0,iy-2),pix(ix+1,iy-2),pix(ix-1,iy-2))
 
      if (omax.gt.pix(ix,iy)) then ! if this pix isn't the brightest
         x = ix                    ! in its 5x5 box, then do something
         y = iy                    ! else, with only the inner 3x3 box
         call find_xyz09(x,y,z,s,pix,psf)
         return
         endif
 
c     do i = 1, 5
c     do j = 1, 5
      do i = 2, 4
      do j = 2, 4
         pixarr(i,j) = pix(ix+i-3,iy+j-3) - s
         wgtarr(i,j) = 1/max(pix(ix+i-3,iy+j-3),s)
         enddo
         enddo
 
      dx0 = 0.0
      dy0 = 0.0
      do NIT = 0, 5
         if (NIT.eq.0) scl = 0.100
         if (NIT.eq.1) scl = 0.100
         if (NIT.eq.2) scl = 0.030
         if (NIT.eq.3) scl = 0.010
         if (NIT.eq.4) scl = 0.003
         if (NIT.eq.5) scl = 0.001
         qofdx   = 0
         qofdy   = 0
         qofflxa = 0
         qofflxw = 0
         qofflxp = 0
         qofmin = 9.0e9
         do i = 01, 11, 01
         do j = 01, 11, 01
            dx = dx0 + (i-6)*scl
            dy = dy0 + (j-6)*scl
            sumf  = 0.
            sump  = 0.
            sumff = 0.
            sumpf = 0.
c           do ii = 1, 5
c           do jj = 1, 5
            do ii = 2, 4
            do jj = 2, 4
               ddx = ii-3-dx
               ddy = jj-3-dy
               ddd = sqrt(ddx**2+ddy**2)
               psfarr(ii,jj) = rpsf_phot(ddx,ddy,psf)
               sumff = sumff + psfarr(ii,jj)*psfarr(ii,jj)
               sumpf = sumpf + pixarr(ii,jj)*psfarr(ii,jj)
               sumf  = sumf  + psfarr(ii,jj)
               sump  = sump  + pixarr(ii,jj)
               enddo
               enddo
            flx = sumpf/sumff
            flx = sump/sumf   ! just use aperture photy ; it's essentially
            qofsum = 0.       ! equiv to noise-weighted PSF-fitting
            wgtsum = 0.       ! for an assumed center
c           do ii = 1, 5
c           do jj = 1, 5
            do ii = 2, 4
            do jj = 2, 4
               dz = (pixarr(ii,jj)-flx*psfarr(ii,jj))
               qofsum = qofsum + wgtarr(ii,jj)*dz**2
               wgtsum = wgtsum + wgtarr(ii,jj)
               enddo
               enddo
            qofsum = qofsum / wgtsum     ! it is actually important not to give
            if (qofsum.lt.qofmin) then   ! the brightest pixels too much weight
               qofmin = qofsum
               qofdx  = dx
               qofdy  = dy
               qofflxa = sump /sumf
               qofflxw = sumpf/sumff
               qofminpct = 0.
c              do ii = 1, 5
c              do jj = 1, 5
               do ii = 2, 4
               do jj = 2, 4
                  dz = (pixarr(ii,jj)-flx*psfarr(ii,jj))
                  qofminpct = qofminpct + abs(dz)/flx
                  enddo
                  enddo
               endif
            enddo
            enddo
        dx0 = qofdx
        dy0 = qofdy
        enddo!NIT
 
      z = qofflxw
      x = ix + dx0
      y = iy + dy0
 
      if (z.lt.1) then ! if this routine didn't arrive
         x = ix        ! at an acceptable flux, try something
         y = iy        ! more simple
         call find_xyz25(x,y,z,s,pix,psf)
         endif
 
      return
      end
 
 
 
c----------------------------------------------
c
c This routine will solve for xyz and sky all from
c the inner 5x5 pixels; it's a bit more robust than
c the plain find_xyz routine, since it doesn't constrain
c sky
c
       subroutine find_xyz25(x,y,z,s,pix,psf)
       implicit none
       real*8 x,y,z,s
       real   pix(_PXDIMX_,_PXDIMY_)
       real   psf(101,101)
 
       real   pixl(5,5)
       real   psfl(5,5)
       integer ix, iy
       real   dx, dy, xc, yc
       real   rpsf_phot
       real   su, zu, ru
       real   rumax
       real   ddx, ddy
       integer i,j
       real   xc0, yc0
       real   SIZE
       integer NIT
 
       integer idx, idy
 
       ix = int(x+0.5)
       iy = int(y+0.5)
 
       do NIT = 1, 9
          if (NIT.eq.1) SIZE = 0.500
          if (NIT.eq.2) SIZE = 0.250
          if (NIT.eq.3) SIZE = 0.150
          if (NIT.eq.4) SIZE = 0.100
          if (NIT.eq.5) SIZE = 0.050
          if (NIT.eq.6) SIZE = 0.020
          if (NIT.eq.7) SIZE = 0.010
          if (NIT.eq.8) SIZE = 0.005
          if (NIT.eq.9) SIZE = 0.003
          xc0 = x
          yc0 = y
          rumax = 0.
          do idx = -2, 2
          do idy = -2, 2
             dx = idx*SIZE/2.0
             dy = idy*SIZE/2.0
             xc = xc0 + dx
             yc = yc0 + dy
             do i = 1, 5
             do j = 1, 5
                pixl(i,j) = pix(ix+i-3,iy+j-3)
                ddx = ix+i-3-xc
                ddy = iy+j-3-yc
                psfl(i,j) = rpsf_phot(ddx,ddy,psf)
                enddo
                enddo
             call LINFITX(psfl,pixl,25,su,zu,ru)
             if (ru.gt.rumax) then
                x = xc
                y = yc
                z = zu
                s = su
                rumax = ru
                endif
             enddo
             enddo
          enddo!NIT
 
      if (z.lt.1) then
         x = ix
         y = iy
         call find_xyz09(x,y,z,s,pix,psf)
         endif
 
      return
      end
 
 
c----------------------------------------------
c
c This routine will solve for xyz and sky all from
c the inner 3x3 pixels; it's even more robust than
c find_xyz25, since the central pixel is definitivelly
c brighter than all the others.
c
      subroutine find_xyz09(x,y,z,s,pix,psf)
      implicit none
      real*8 x,y,z,s
      real   pix(_PXDIMX_,_PXDIMY_)
      real   psf(101,101)
 
      real   ptot, xtot, ytot, ftot
 
      integer xlist(4)
      integer ylist(4)
      real    plist(4)
 
      real    ddx, ddy
      real    smin
      real    rpsf_phot
      integer ix, iy, ii, jj
 
 
      ix = int(x+0.5)
      iy = int(y+0.5)
 
      smin = min(pix(ix  ,iy  ),
     .           pix(ix-1,iy  ),
     .           pix(ix+1,iy  ),
     .           pix(ix  ,iy-1),
     .           pix(ix  ,iy+1),
     .           pix(ix-1,iy-1),
     .           pix(ix-1,iy+1),
     .           pix(ix+1,iy-1),
     .           pix(ix+1,iy+1))
 
      xtot = 0.
      ytot = 0.
      ptot = 0.
      do ii = ix-1, ix+1
      do jj = iy-1, iy+1
         xtot = xtot + ii*(pix(ii,jj)-smin)
         ytot = ytot + jj*(pix(ii,jj)-smin)
         ptot = ptot +    (pix(ii,jj)-smin)
         enddo
         enddo
      x = xtot/ptot
      y = ytot/ptot
 
      ptot = 0.
      ftot = 0.
      do ii = ix-1, ix+1
      do jj = iy-1, iy+1
         ddx = ii-x
         ddy = jj-y
         if (abs(ddx).lt.1.0.and.abs(ddy).lt.1.0) then
             ptot = ptot + pix(ii,jj)-s
             ftot = ftot + rpsf_phot(ddx,ddy,psf)
             endif
         enddo
         enddo
      z = ptot/ftot
 
      return
      end
 
 
 
 
c----------------------------------------------
c
c This routine will solve for xyz and sky all from
c the inner 3x3 pixels; it's even more robust than
c find_xyz25, since the central pixel is definitivelly
c brighter than all the others.
c
       subroutine find_xyz09x(x,y,z,s,pix,psf)
       implicit none
       real*8 x,y,z,s
       real   pix(_PXDIMX_,_PXDIMY_)
       real   psf(101,101)
 
       real   pixl(3,3)
       real   psfl(3,3)
       integer ix, iy
       real   dx, dy, xc, yc
       real   rpsf_phot
       real   su, zu, ru
       real   rumax
       real   ddx, ddy
       integer i,j
       real   xc0, yc0
       real   SIZE
       integer NIT
       integer idx, idy
 
       ix = int(x+0.5)
       iy = int(y+0.5)
 
       do NIT = 1, 9
          if (NIT.eq.1) SIZE = 0.500
          if (NIT.eq.2) SIZE = 0.250
          if (NIT.eq.3) SIZE = 0.150
          if (NIT.eq.4) SIZE = 0.100
          if (NIT.eq.5) SIZE = 0.050
          if (NIT.eq.6) SIZE = 0.020
          if (NIT.eq.7) SIZE = 0.010
          if (NIT.eq.8) SIZE = 0.005
          if (NIT.eq.9) SIZE = 0.003
          xc0 = x
          yc0 = y
          rumax = 0.
          do idx = -2, 2
          do idy = -2, 2
             dx = idx*SIZE/2.0
             dy = idy*SIZE/2.0
             xc = xc0 + dx
             yc = yc0 + dy
             do i = 1, 3
             do j = 1, 3
                pixl(i,j) = pix(ix+i-2,iy+j-2)
                ddx = ix+i-3-xc
                ddy = iy+j-3-yc
                psfl(i,j) = rpsf_phot(ddx,ddy,psf)
                enddo
                enddo
             call LINFITX(psfl,pixl,09,su,zu,ru)
             if (ru.gt.rumax) then
                x = xc
                y = yc
                z = zu
                s = su
                rumax = ru
                endif
             enddo
             enddo
          enddo!NIT
 
      return
      end
 
 
c--------------------------------------------
c
c Least squares solution for a line from BEVINGTON
c
c Generally:  pix(i) = z*psf(i) + sky
c
      SUBROUTINE LINFITX(X,Y,NPTS,A,B,R)
      IMPLICIT NONE
      REAL X(*), Y(*)
      INTEGER NPTS
 
      REAL A, B, R
 
      REAL SUM, SUMX, SUMY, SUMX2, SUMXY, SUMY2
      REAL DELTA
      INTEGER I
 
   11 SUM = 0
      SUMX = 0
      SUMY = 0
      SUMX2 = 0
      SUMXY = 0
      SUMY2 = 0
   21 DO 59  I = 1, NPTS
           SUM   = SUM   + 1
           SUMX  = SUMX  + X(I)
           SUMY  = SUMY  + Y(I)
           SUMX2 = SUMX2 + X(I)*X(I)
           SUMXY = SUMXY + X(I)*Y(I)
           SUMY2 = SUMY2 + Y(I)*Y(I)
   59 CONTINUE
 
   51 DELTA = SUM*SUMX2 - SUMX*SUMX
      A = (SUMX2*SUMY - SUMX*SUMXY) / DELTA
   53 B = (SUMXY*SUM  - SUMX*SUMY ) / DELTA
      R = 0.00
   71 IF (DELTA*(SUM*SUMY2 - SUMY*SUMY).GE.0) THEN
          R = (SUM*SUMXY - SUMX*SUMY) /
     C        SQRT(DELTA*(SUM*SUMY2 - SUMY*SUMY))
          ENDIF
 
 
      RETURN
      END
 
 
c-----------------------------------------------------
c
c This routine will find the position and flux for a
c saturated star by identifying the closest unsaturated
c pixels and fitting a PSF to them, solving for flux
c and position.
c
      subroutine find_xyz_SAT(x,y,z,s,pix,psf)
      implicit none
 
      real*8 x,y,z,s
      real   pix(_PXDIMX_,_PXDIMY_)
      real   psf(101,101)
 
      integer i0, j0
      integer i, j
      real    plist(9999)
      real    flist(9999)
      real    qlist(9999)
      integer ilist(9999)
      integer jlist(9999)
 
      integer NT, NL, NLs
      real rpsf_phot
      real mode_sky
 
      real dx, dy, r, RMAX
      real QOFMIN, QOFDX, QOFDY, QOFFLX, QOF
      real*8 PTOT, FTOT, FLX
      integer NNN
      common /nqq/nnn
      data NNN/0/
 
      integer idx, idy
 
      i0 = int(x+0.5)
      j0 = int(y+0.5)
 
      s = mode_sky(i0,j0,15,25,pix)
 
      RMAX = 2.5
  1   continue
      NT = 0
      NL = 0
      RMAX = RMAX + 1
      do i = i0-int(RMAX+1),i0+int(RMAX+1)
      do j = j0-int(RMAX+1),j0+int(RMAX+1)
         r = sqrt(((i-i0)**2+(j-j0)**2)*1.0)
         if (r.ge.RMAX) goto 2
         if (i.lt.2.or.i.gt.(_PXDIMX_-1)) goto 2
         if (j.lt.2.or.j.gt.(_PXDIMY_-1)) goto 2
         NT = NT + 1
         if (pix(i,j  ).gt._HIFLAG_) goto 2
         if (pix(i,j+1).gt._HIFLAG_) goto 2
         if (pix(i,j-1).gt._HIFLAG_) goto 2
         NL = NL + 1
         plist(NL) = pix(i,j)-s
         ilist(NL) = i
         jlist(NL) = j
 2       continue
         enddo
         enddo
      if (NL.lt.0.75*NT) goto 1
 
      NLs = NL
 
      QOFMIN = 9.9e9
      QOFDX  = 0.0
      QOFDY  = 0.0
      QOFFLX = 0.0
      do idx = -20, 20
      do idy = -20, 20
         dx = idx*0.05
         dy = idy*0.05
         PTOT = 0.
         FTOT = 0.
         do NL = 1, NLs
            flist(NL) = rpsf_phot(ilist(NL)-i0-dx,
     .                            jlist(NL)-j0-dy,psf)
            PTOT = PTOT + plist(NL)
            FTOT = FTOT + flist(NL)
            enddo
         FLX = PTOT/FTOT
         QOF = 0.
         do NL = 1, NLs
            QOF = QOF + abs(plist(NL)-FLX*flist(NL))
           enddo
         if (QOF.lt.QOFMIN) then
            QOFMIN = QOF
            QOFDX  = DX
            QOFDY  = DY
            QOFFLX = FLX
            do NL = 1, NLs
               qlist(NL) = flist(NL)
               enddo
            endif
         enddo
         enddo
 
      x = x + QOFDX
      y = y + QOFDY
      z = QOFFLX
 
      return
      end
 
 
 
 
c----------------------------------------------
c this function orders the pixels in increasing
c distance from a central pixel.
c
      integer function oi(m)
      implicit none
      integer m
 
      integer qm, fm
      integer im
      integer mm
      integer oim, ojm
      integer nm
 
      integer oil(91)
      data oil / 0,1,1,2,2,1,2,3,3,1,3,2,4,4,
     .               1,3,4,2,3,5,4,5,1,5,2,4,
     .           3,5,6,1,6,6,2,5,4,3,6,7,5,1,
     .               7,6,4,2,7,7,3,6,5,8,1,4,
     .           8,7,8,2,6,8,3,7,5,4,8,9,1,9,
     .                 9,7,2,6,5,8,3,9,4,9,7,
     .           9,5,8,6,8,7,8,9,6,9,7,9,8,9 /
 
      integer ojl(91)
      data ojl / 0,0,1,0,1,2,2,0,1,3,2,3,0,1,4,
     .                   3,2,4,4,0,3,1,5,2,5,4,
     .           5,3,0,6,1,2,6,4,5,6,3,0,5,7,1,
     .                   4,6,7,2,3,7,5,6,0,8,7,
     .           1,4,2,8,6,3,8,5,7,8,4,0,9,1,2,
     .                     6,9,7,8,5,9,3,9,4,7,
     .             5,9,6,8,7,8,8,6,9,7,9,8,9,9 /
 
 
      qm = m+2 - 4*int((m+2)/4)
      fm = int((m+6)/4)
      if (fm.le.91) then
         if (qm.eq.0) oi =  oil(fm)
         if (qm.eq.1) oi = -oil(fm)
         if (qm.eq.2) oi = -ojl(fm)
         if (qm.eq.3) oi =  ojl(fm)
         return
         endif
 
      im = (sqrt(1.*m-1)-1)/2
      mm = (2*im+1)**2+1
      nm =  (m-mm)/4
      oim = nm-im
      ojm = im
 
      if (qm.eq.0) oi =  oim
      if (qm.eq.1) oi = -oim
      if (qm.eq.2) oi = -ojm
      if (qm.eq.3) oi =  ojm
 
      return
      end
 
c----------------------------------------------------------------
 
      integer function oj(m)
      implicit none
      integer m
 
      integer qm, fm
      integer im, mm, nm, oim, ojm
 
      integer oil(91)
      data oil / 0,1,1,2,2,1,2,3,3,1,3,2,4,4,
     .               1,3,4,2,3,5,4,5,1,5,2,4,
     .           3,5,6,1,6,6,2,5,4,3,6,7,5,1,
     .               7,6,4,2,7,7,3,6,5,8,1,4,
     .           8,7,8,2,6,8,3,7,5,4,8,9,1,9,
     .                 9,7,2,6,5,8,3,9,4,9,7,
     .           9,5,8,6,8,7,8,9,6,9,7,9,8,9 /
 
      integer ojl(91)
      data ojl / 0,0,1,0,1,2,2,0,1,3,2,3,0,1,4,
     .                   3,2,4,4,0,3,1,5,2,5,4,
     .           5,3,0,6,1,2,6,4,5,6,3,0,5,7,1,
     .                   4,6,7,2,3,7,5,6,0,8,7,
     .           1,4,2,8,6,3,8,5,7,8,4,0,9,1,2,
     .                     6,9,7,8,5,9,3,9,4,7,
     .             5,9,6,8,7,8,8,6,9,7,9,8,9,9 /
 
      qm = m+2 - 4*int((m+2)/4)
      fm = int((m+6)/4)
      if (fm.le.91) then
         if (qm.eq.0) oj =  ojl(fm)
         if (qm.eq.1) oj = -ojl(fm)
         if (qm.eq.2) oj =  oil(fm)
         if (qm.eq.3) oj = -oil(fm)
         return
         endif
 
      im = (sqrt(1.*m-1)-1)/2
      mm = (2*im+1)**2+1
      nm =  (m-mm)/4
      oim = nm-im
      ojm = im
 
      if (qm.eq.0) oj =  ojm
      if (qm.eq.1) oj = -ojm
      if (qm.eq.2) oj =  oim
      if (qm.eq.3) oj = -oim
 
      return
      end
 
c----------------------------------------------------------------
 
      real function orm(m)
      implicit none
      integer m
 
      integer oi, oj
 
      orm = sqrt(1.*(oi(m)**2 + oj(m)**2))
 
      return
      end
 
c----------------------------------------------------------------
c
c this routine performs aperture photometry, including those
c pixels with r<rap and taking sky from a mode of the pixels
c between r>ri and r<ro.
c
      real function apphot(i,j,rap,ri,ro,pix,sky)
      implicit none
 
      integer i, j
      real    rap
      integer ri, ro
      real pix(_PXDIMX_,_PXDIMY_)
      real    sky
 
 
      integer ii, jj
      real mode_sky
      integer ntot
      real    ftot
 
      if (ri.lt.99) sky = mode_sky(i,j,ri,ro,pix)
 
      ntot = 0
      ftot = 0
      do ii = i-int(rap+1),i+int(rap+1)
      do jj = j-int(rap+1),j+int(rap+1)
         if (ii.lt.      01) goto 1
         if (jj.lt.      01) goto 1
         if (ii.gt._PXDIMX_) goto 1
         if (jj.gt._PXDIMY_) goto 1
         if ((ii-i)**2+(jj-j)**2.gt.rap**2) goto 1
         ftot = ftot + pix(ii,jj)-sky
         ntot = ntot + 1
 1       continue
         enddo
         enddo
 
      apphot = ftot
 
      return
      end
 
 
 
 
c--------------------------------------
c
c this will evaluate a PSF at a given (dx,dy)
c offset; for the regions within 4 pixels of
c the center, it uses bi-cubic interpolation
c
      real function rpsf_phot(x,y,psf)
      implicit none
      real x, y
      real psf(101,101)
 
 
      real    rx, ry
      integer ix, iy   !     3   4
      real    fx, fy   !    *1*  2
 
      real    dd
 
      real A1, B1, C1, D1, E1, F1, V1
      real A2, B2, C2, D2, E2, F2, V2
      real A3, B3, C3, D3, E3, F3, V3
      real A4, B4, C4, D4, E4, F4, V4
 
c      real rpsf_phut
c
c      rpsf_phot = rpsf_phut(x,y)
c
c      if (.true.) return
 
 
      rx = 51 + x*4
      ry = 51 + y*4
      ix = int(rx)
      iy = int(ry)
      fx = rx-ix
      fy = ry-iy
 
      dd = sqrt(x**2+y**2)
 
 
      rpsf_phot = 0.
      if (dd.gt.12.0) return
 
      if (dd.gt.4.0) then
         rpsf_phot = (1-fx)*(1-fy)*psf(ix  ,iy  )
     .             + ( fx )*(1-fy)*psf(ix+1,iy  )
     .             + (1-fx)*( fy )*psf(ix  ,iy+1)
     .             + ( fx )*( fy )*psf(ix+1,iy+1)
         return
         endif
 
 
      A1 =  psf(ix  ,iy  )
      B1 = (psf(ix+1,iy  )-psf(ix-1,iy  ))/2
      C1 = (psf(ix  ,iy+1)-psf(ix  ,iy-1))/2
      D1 = (psf(ix+1,iy  )+psf(ix-1,iy  )-2*A1)/2
      F1 = (psf(ix  ,iy+1)+psf(ix  ,iy-1)-2*A1)/2
      E1 = (psf(ix+1,iy+1)-A1)
 
      A2 =  psf(ix+1,iy  )
      B2 = (psf(ix+2,iy  )-psf(ix  ,iy  ))/2
      C2 = (psf(ix+1,iy+1)-psf(ix+1,iy-1))/2
      D2 = (psf(ix+2,iy  )+psf(ix  ,iy  )-2*A2)/2
      F2 = (psf(ix+1,iy+1)+psf(ix+1,iy-1)-2*A2)/2
      E2 =-(psf(ix  ,iy+1)-A2)
 
      A3 =  psf(ix  ,iy+1)
      B3 = (psf(ix+1,iy+1)-psf(ix-1,iy+1))/2
      C3 = (psf(ix  ,iy+2)-psf(ix  ,iy  ))/2
      D3 = (psf(ix+1,iy+1)+psf(ix-1,iy+1)-2*A3)/2
      F3 = (psf(ix  ,iy+2)+psf(ix  ,iy  )-2*A3)/2
      E3 =-(psf(ix+1,iy  )-A3)
 
      A4 =  psf(ix+1,iy+1)
      B4 = (psf(ix+2,iy+1)-psf(ix  ,iy+1))/2
      C4 = (psf(ix+1,iy+2)-psf(ix+1,iy  ))/2
      D4 = (psf(ix+2,iy+1)+psf(ix  ,iy+1)-2*A4)/2
      F4 = (psf(ix+1,iy+2)+psf(ix+1,iy  )-2*A4)/2
      E4 = (psf(ix  ,iy  )-A4)
 
 
      V1 = A1
     .   + B1*( fx )
     .   + C1*( fy )
     .   + D1*( fx )**2
     .   + E1*( fx )*( fy )
     .   + F1*( fy )**2
 
      V2 = A2
     .   + B2*(fx-1)
     .   + C2*( fy )
     .   + D2*(fx-1)**2
     .   + E2*(fx-1)*( fy )
     .   + F2*( fy )**2
 
      V3 = A3
     .   + B3*( fx )
     .   + C3*(fy-1)
     .   + D3*( fx )**2
     .   + E3*( fx )*(fy-1)
     .   + F3*(fy-1)**2
 
      V4 = A4
     .   + B4*(fx-1)
     .   + C4*(fy-1)
     .   + D4*(fx-1)**2
     .   + E4*(fx-1)*(fy-1)
     .   + F4*(fy-1)**2
 
      rpsf_phot = (1-fx)*(1-fy)*V1
     .          + ( fx )*(1-fy)*V2
     .          + (1-fx)*( fy )*V3
     .          + ( fx )*( fy )*V4
 
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/writfits_r4.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------
c
c this just writes a real*4 fits image
c
 
      subroutine writfits_r4(FILE,pix,PXDIMX,PXDIMY)
      implicit none
 
      character*(*) FILE
      integer PXDIMX,PXDIMY
      real    pix(PXDIMX,PXDIMY)
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffb,buffc)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer k
 
      character*80 FILEU
      character*70 HDR(25)
      common/HDR/HDR
 
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      open(10,file=FILEU,status='unknown',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      do k = 00, 34
         write(buffc(k*80+1:k*80+80),'(80x)')
         enddo
 
      write(10,rec=i,iostat=ios) buffc
 
      write(buffc( 0*80+1: 1*80),'(''SIMPLE  =                    T'')')
      write(buffc( 1*80+1: 2*80),'(''BITPIX  =                  -32'')')
      write(buffc( 2*80+1: 3*80),'(''NAXIS   = '',8x,i12)') 2
      write(buffc( 3*80+1: 4*80),'(''NAXIS1  = '',8x,i12)') PXDIMX
      write(buffc( 4*80+1: 5*80),'(''NAXIS2  = '',8x,i12)') PXDIMY
      write(buffc( 5*80+1: 6*80),'(''DATATYPE= '',9a)')
     .                          " 'REAL*4' "
      write(buffc(07*80+1:08*80),'(''COMMENT   '',a05)') '     '
      write(buffc(08*80+1:09*80),'(''COMMENT   '',a05)') '     '
      write(buffc(09*80+1:10*80),'(''CRPIX1  = '',a70)') HDR(01)
      write(buffc(10*80+1:11*80),'(''CRPIX2  = '',a70)') HDR(02)
      write(buffc(11*80+1:12*80),'(''CRVAL1  = '',a70)') HDR(03)
      write(buffc(12*80+1:13*80),'(''CRVAL2  = '',a70)') HDR(04)
      write(buffc(13*80+1:14*80),'(''CTYPE1  = '',a70)') HDR(05)
      write(buffc(14*80+1:15*80),'(''CTYPE2  = '',a70)') HDR(06)
      write(buffc(15*80+1:16*80),'(''CD1_1   = '',a70)') HDR(07)
      write(buffc(16*80+1:17*80),'(''CD1_2   = '',a70)') HDR(08)
      write(buffc(17*80+1:18*80),'(''CD2_1   = '',a70)') HDR(09)
      write(buffc(18*80+1:19*80),'(''CD2_2   = '',a70)') HDR(10)
      write(buffc(19*80+1:20*80),'(''ORIENTAT= '',a70)') HDR(11)
      write(buffc(20*80+1:21*80),'(''PA_APER = '',a70)') HDR(12)
      write(buffc(21*80+1:22*80),'(''PA_V3   = '',a70)') HDR(13)
      write(buffc(22*80+1:23*80),'(''DATE-OBS= '',a70)') HDR(14)
      write(buffc(23*80+1:24*80),'(''TIME-OBS= '',a70)') HDR(15)
      write(buffc(24*80+1:25*80),'(''EXPTIME = '',a70)') HDR(16)
      write(buffc(25*80+1:26*80),'(''ROOTNAME= '',a70)') HDR(17)
      write(buffc(26*80+1:27*80),'(''TARGNAME= '',a70)') HDR(18)
      write(buffc(27*80+1:28*80),'(''RA_TARG = '',a70)') HDR(19)
      write(buffc(28*80+1:29*80),'(''DEC_TARG= '',a70)') HDR(20)
      write(buffc(29*80+1:30*80),'(''PROPOSID= '',a70)') HDR(21)
      write(buffc(30*80+1:31*80),'(''FILTER1 = '',a70)') HDR(22)
      write(buffc(31*80+1:32*80),'(''FILTER2 = '',a70)') HDR(23)
      write(buffc(33*80+1:34*80),'(''VAFACTOR= '',a70)') HDR(24)
      write(buffc(32*80+1:33*80),'(''CCDGAIN = '',a70)') HDR(25)
      write(buffc(33*80+1:34*80),'(''COMMENT   '',a05)') '     '
      write(buffc(34*80+1:35*80),'(''COMMENT   '',a05)') '     '
      write(buffc(35*80+1:36*80),'(''END       '')')
 
      write(10,rec=i,iostat=ios) buffc
 
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      nbper  = 4*PXDIMX*PXDIMY
      npt    =   PXDIMX*PXDIMY
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      do i = i1, i2, 1
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         call pix2buff_r4(buffb,pix,np1,npt)
         write(10,rec=i,iostat=ios) buffc
         enddo
 
      close(10)
 
      return
 
 900  continue
      print*,'writfits_r4.f ERROR'
      print*,'   FILEU: ',FILEU
      stop
 
      end
 
c-------------------------------------------------------
c
c
      subroutine pix2buff_r4(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      real*4 pix(*)
      integer n1,nt
 
      byte b(4)
      real*4 r
      equivalence(r,b)
 
      integer i, npu, nbu
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (npu.ge.1.and.npu.le.nt) r = pix(npu)
         if (.not.(_LINUX_)) then
            buff(nbu+1) = b(1)
            buff(nbu+2) = b(2)
            buff(nbu+3) = b(3)
            buff(nbu+4) = b(4)
            endif
         if ((_LINUX_)) then
            buff(nbu+1) = b(4)
            buff(nbu+2) = b(3)
            buff(nbu+3) = b(2)
            buff(nbu+4) = b(1)
            endif
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/writfits_i4.f"
c****                                         
c*********************************************
                                              
 
c------------------------------------------------------
c
c this routine will write an integer*4 array into
c a 2-dimensional fits image
c
      subroutine writfits_i4(FILE,pix,PXDIMX,PXDIMY)
      implicit none
 
      integer PXDIMX,PXDIMY
      character*(*) FILE
      integer*4 pix(PXDIMX,PXDIMY)
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios
 
      integer ii, jj
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
 
      logical DIAG
      data DIAG/.false./
 
      character*70 HDR(25)
      common/HDR/HDR
      character*80 FILEU
 
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      open(10,file=FILEU,
     .     status='unknown',
     .     err   =900,
     .     recl  =2880,
     .     form  ='UNFORMATTED',
     .     access='DIRECT')
 
      write(buffc( 0*80+1: 1*80),'(''SIMPLE  = T    '')')
      write(buffc( 1*80+1: 2*80),'(''BITPIX  =  32  '')')
      write(buffc( 2*80+1: 3*80),'(''NAXIS   ='',i12)') 2
      write(buffc( 3*80+1: 4*80),'(''NAXIS1  ='',i12)') PXDIMX
      write(buffc( 4*80+1: 5*80),'(''NAXIS2  ='',i12)') PXDIMY
      write(buffc( 5*80+1: 6*80),'(''DATATYPE='',9a)')
     .                          ' ''INTEGER*4'''
      write(buffc( 6*80+1: 7*80),'(''DATE    ='',11a)')
     .                          ' ''00/00/00'''
      write(buffc( 7*80+1: 8*80),'(''BSCALE  ='',i12)') 00001
      write(buffc( 8*80+1: 9*80),'(''BZERO   ='',i12)') 00000
      write(buffc(09*80+1:10*80),'(''CRPIX1  ='',a20)') HDR(01)
      write(buffc(10*80+1:11*80),'(''CRPIX2  ='',a20)') HDR(02)
      write(buffc(11*80+1:12*80),'(''CRVAL1  ='',a20)') HDR(03)
      write(buffc(12*80+1:13*80),'(''CRVAL2  ='',a20)') HDR(04)
      write(buffc(13*80+1:14*80),'(''CTYPE1  ='',a20)') HDR(05)
      write(buffc(14*80+1:15*80),'(''CTYPE2  ='',a20)') HDR(06)
      write(buffc(15*80+1:16*80),'(''CD1_1   ='',a20)') HDR(07)
      write(buffc(16*80+1:17*80),'(''CD1_2   ='',a20)') HDR(08)
      write(buffc(17*80+1:18*80),'(''CD2_1   ='',a20)') HDR(09)
      write(buffc(18*80+1:19*80),'(''CD2_2   ='',a20)') HDR(10)
      write(buffc(19*80+1:20*80),'(''ORIENTAT='',a20)') HDR(11)
      write(buffc(20*80+1:21*80),'(''PA_APER ='',a20)') HDR(12)
      write(buffc(21*80+1:22*80),'(''PA_V3   ='',a20)') HDR(13)
      write(buffc(22*80+1:23*80),'(''DATE-OBS='',a20)') HDR(14)
      write(buffc(23*80+1:24*80),'(''TIME-OBS='',a20)') HDR(15)
      write(buffc(24*80+1:25*80),'(''EXPTIME ='',a20)') HDR(16)
      write(buffc(25*80+1:26*80),'(''ROOTNAME='',a20)') HDR(17)
      write(buffc(26*80+1:27*80),'(''TARGNAME='',a20)') HDR(18)
      write(buffc(27*80+1:28*80),'(''RA_TARG ='',a20)') HDR(19)
      write(buffc(28*80+1:29*80),'(''DEC_TARG='',a20)') HDR(20)
      write(buffc(29*80+1:30*80),'(''PROPOSID='',a20)') HDR(21)
      write(buffc(30*80+1:31*80),'(''FILTER1 ='',a20)') HDR(22)
      write(buffc(31*80+1:32*80),'(''FILTER2 ='',a20)') HDR(23)
      write(buffc(33*80+1:34*80),'(''VAFACTOR='',a20)') HDR(24)
      write(buffc(32*80+1:33*80),'(''CCDGAIN ='',a20)') HDR(25)
      write(buffc(34*80+1:35*80),'(''COMMENT  '')')
      write(buffc(35*80+1:36*80),'(''END      '')')
 
      i = 1
      write(10,rec=i,iostat=ios) buffc
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      nbper  = 4*PXDIMX*PXDIMY
      npt    =   PXDIMX*PXDIMY
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      if (DIAG) then
         print*,'----->      i1: ',i1
         print*,'----->      i2: ',i2
         print*,'----->  PXDIMX: ',PXDIMX
         print*,'----->  PXDIMY: ',PXDIMY
         print*,'----->   nbper: ',nbper
         print*,'----->     npt: ',npt
         print*,'----->  C'
         endif
 
      do ii = 1, PXDIMX
      do jj = 1, PXDIMY
         pix(ii,jj) = pix(ii,jj)
         enddo
         enddo
 
      do i = i1, i2, 1
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         call pix2buff_i4(buffb,pix,np1,npt)
         write(10,rec=i,iostat=ios) buffc
         enddo
 
      close(10)
 
      return
 
 900  continue
      print*,' '
      print*,'WRITFITS_i4 ERROR: '
      print*,' COULD NOT OPEN FILE: '
      print*,' FILE: ',FILEU
      print*,' '
      stop
 
      end
 
 
c
c
c
      subroutine pix2buff_i4(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      integer*4 pix(*)
      integer n1,nt
 
      byte b(4)
      integer ii
      equivalence(ii,b)
 
      integer i, npu, nbu
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (npu.ge.1.and.npu.le.nt) ii = pix(npu)
         if (.not.(_LINUX_)) then
            buff(nbu+1) = b(1)
            buff(nbu+2) = b(2)
            buff(nbu+3) = b(3)
            buff(nbu+4) = b(4)
            endif
         if ((_LINUX_)) then
            buff(nbu+1) = b(4)
            buff(nbu+2) = b(3)
            buff(nbu+3) = b(2)
            buff(nbu+4) = b(1)
            endif
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_r4.f"
c****                                         
c*********************************************
                                              
 
c--------------------------------------------------
c
c this just reads in an r4 fits image
c
 
      subroutine readfits_r4(FILE,pix,NDIMX,NDIMY)
      implicit none
 
      character*(*) FILE
      integer NDIMX,NDIMY
      real    pix(NDIMX,NDIMY)
 
      character*199 FILEU
      character*070 INFO(10)
      common / fitsinfo / INFO
 
      integer naxes
      integer laxis(3)
      common/laxis3_/laxis
 
      character*8  field
      character*70 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer j
 
      character*2880 buffc
      byte   buffb(2880)
      equivalence (buffc,buffb)
 
      real*4 buffr(0720)
      integer ii,nn,nx,ny
      integer nxx, nyy
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      logical DIAG
      data DIAG /.false./
 
      character*70 HDR(25)
      common/HDR/HDR
 
      logical extend_tf
 
      FILEU = FILE
      do i = 195,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      do i = 1, 25
         HDR(i) = ' '
         enddo
 
 
      if (DIAG) then
         print*,'enter readfits_r4...'
         print*,'FILE: ',FILE(1:60)
         endif
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened'
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
 
      extend_tf = .false.
 
      i = 0
      nread = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buffc
      if (DIAG) print*,'READREC: ',i
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+10:k*80+79)
         if (field.eq.'EXTEND  ') read(stream,*) extend_tf
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(1) = stream
         if (field.eq.'FILTNAM1') INFO(2) = stream
         if (field.eq.'FILENAME') INFO(3) = stream
         if (field.eq.'DATE-OBS') INFO(4) = stream
         if (field.eq.'TIME-OBS') INFO(5) = stream
         if (field.eq.'DEC_TARG') INFO(6) = stream
         if (field.eq.'RA_TARG ') INFO(7) = stream
         if (field.eq.'PA_V3   ') INFO(8) = stream
         if (field.eq.'PROPOSID') INFO(9) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER1 ') HDR(22) = stream
         if (field.eq.'FILTER2 ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
 
         if (field.eq.'END     ') goto 101
         enddo
      goto 100
 101  continue
 
      nread = nread + 1
 
      if (extend_tf.and.NREAD.eq.1) goto 100 ! read the first extension...
 
      if (DIAG) then
         print*,'----------------------------------------'
         print*,'  NREAD: ',nread
         print*,'NEXTEND: ',nextend
         print*,'  NAXIS: ',naxes
         print*,'  LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,' BITPIX: ',bitpix
         print*,' BSCALE: ',bscale
         print*,'  BZERO: ',bzero
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      nbper  = 4*laxis(1)*laxis(2)
      npt    = laxis(1)*laxis(2)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      if (BITPIX.ne.-32) then
         print*,'BITPIX.ne.-32... unreal!'
         stop
         endif
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         call buff2pix_r4(buffb,buffr,0001,0720)
         do ii = 001, 720
            nn = np1 + (ii-1)
            ny = 1 + (nn-1)/laxis(1)
            nx = nn-(ny-1)*laxis(1)
            if (nx.ge.001.and.nx.le.NDIMX.and.
     .          ny.ge.001.and.ny.le.NDIMY) then
                pix(nx,ny) = buffr(ii)
                nxx = nx
                nyy = ny
                endif
            enddo
         if (ny.gt.NDIMY) goto 899
 
         if (DIAG) write(*,1115) i,np1,np2,npt,nxx,nyy
 1115    format(1x,i8,1x,i10,1x,i10,1x,i10,1x,2i6)
         enddo
 
  899 close(10)
      if (DIAG) write(*,1115) i,np1,np2,npt,nxx,nyy
 
 
 
      return
 
  900 continue
      print*,'                '
      print*,'READFITS ERROR: '
      print*,'                '
      write(*,'(''   could not read in file:   '',80a)') FILEU
      print*,'                '
      stop
 
      end
 
 
 
      subroutine buff2pix_r4(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      real pix(*)
      integer n1,nt
 
      byte b(4)
      real r
      equivalence(r,b)
 
      integer i, npu, nbu
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (.not.(_LINUX_)) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            b(3) = buff(nbu+3)
            b(4) = buff(nbu+4)
            endif
         if ((_LINUX_)) then
            b(4) = buff(nbu+1)
            b(3) = buff(nbu+2)
            b(2) = buff(nbu+3)
            b(1) = buff(nbu+4)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = r
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_i4r.f"
c****                                         
c*********************************************
                                              
 
c--------------------------------------------------------
c
c This routine reads in images stored in integer*4 format...
c
c to use:
c
c   input parameters:
c
c         FILE     is the filename
c         NX,NY    are the dimensions of the image to be
c                  read in
c
c
c  output parameters:
c
c         pix( , ) is an integer*4 image of dimensions NX,NY
c
c
c
      subroutine readfits_i4r(FILE,pix,NX,NY)
      implicit none
 
      integer NX, NY
      character*80 FILE
      real*4 pix(NX,NY)
 
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      integer naxes
      integer laxis(3)
 
      character*8  field
      character*20 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer ii, jj
      integer n
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffb,buffc)
 
      integer*4 ibuff(720)
 
      integer ifirst, i1, i2
      integer j
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      logical LINUX
      data LINUX/.true./
      logical DIAG
      data DIAG /.false./
 
 
      character*70 HDR(25)
      common/HDR/HDR
 
      character*80 FILEU
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      if (DIAG) then
         print*,'enter readfits...'
         print*,'FILE: ',FILE(1:60)
         endif
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened'
 
      bscale = 1
      bzero  = 0
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
      do i = 1, 23
         HDR(i) = ' '
         enddo
 
      i = 0
      nread = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buffc
      if (DIAG) print*,'READREC: ',i
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+31)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(01) = stream
         if (field.eq.'FILTNAM1'.and.stream(1:1).eq.'F')
     .                            INFO(02) = stream
         if (field.eq.'FILTNAM2'.and.stream(1:1).eq.'F')
     .                            INFO(02) = stream
         if (field.eq.'FILTER1 '.and.stream(1:1).eq.'F')
     .                            INFO(02) = stream
         if (field.eq.'FILTER2 '.and.stream(1:1).eq.'F')
     .                            INFO(02) = stream
         if (field.eq.'FILENAME') INFO(03) = stream
         if (field.eq.'DATE-OBS') INFO(04) = stream
         if (field.eq.'TIME-OBS') INFO(05) = stream
         if (field.eq.'DEC_TARG') INFO(06) = stream
         if (field.eq.'RA_TARG ') INFO(07) = stream
         if (field.eq.'PA_V3   ') INFO(08) = stream
         if (field.eq.'PROPOSID') INFO(09) = stream
         if (field.eq.'CCDGAIN ') INFO(10) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER1 ') HDR(22) = stream
         if (field.eq.'FILTER2 ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
 
 
         if (field.eq.'END     ') goto 101
         enddo
      goto 100
 101  continue
 
 
      if (laxis(1).ne.NX.or.laxis(2).ne.NY) then
         print*,'FITS image not the expected dimensions: '
         print*,'    input NX: ',nx
         print*,'    input NY: ',ny
         print*,'              '
         print*,'    laxis(1): ',laxis(1)
         print*,'    laxis(2): ',laxis(2)
         print*,'  '
         stop
         endif
 
      nread = nread + 1
      if (DIAG) then
         print*,'----------------------------------------'
         print*,'  NREAD: ',nread
         print*,'NEXTEND: ',nextend
         print*,'  NAXIS: ',naxes
         print*,'  LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,' BITPIX: ',bitpix
         print*,' BSCALE: ',bscale
         print*,'  BZERO: ',bzero
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      if (BITPIX.ne.32) then
         print*,'                '
         print*,'readfits_i4...: '
         print*,'                '
         print*,'   you called a routine to read in a'
         print*,'   long i*4 image, and the image you'
         print*,'   gave it has BITPIX = ',BITPIX
         print*,'   '
         print*,'   FILE: ',FILE
         print*,'   '
         stop
         endif
 
      nbper = 4*laxis(1)*laxis(2)
      npt   =   laxis(1)*laxis(2)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         np2 = min(np2,npt)
         call buff2pix_i4r(buffb,ibuff,0001,0720)
         do n = np1, np2, 1
            jj = n/NX   + 1
            ii = n-NX*(jj-1)
            pix(ii,jj) = ibuff(n-np1+1)*bscale+bzero
            enddo
         enddo
 
      if (DIAG) then
         print*,'  NBPER: ',nbper
         print*,'  NBYT1: ',nbyte1
         print*,'  NBYT2: ',nbyte2
         print*,' IFIRST: ',ifirst
         print*,'     I1: ',i1
         print*,'     I2: ',i2
         print*,'    NPT: ',NPT
         endif
 
      return
 
 900  continue
      print*,'READFITS_I4R ERROR'
      print*,' FILEU: ',FILEU
      stop
 
      end
 
 
      subroutine buff2pix_i4r(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      integer*4 pix(*)
      integer n1,nt
 
      byte b(4)
      integer ii
      equivalence(ii,b)
 
      integer i, npu, nbu
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (.not.(_LINUX_)) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            b(3) = buff(nbu+3)
            b(4) = buff(nbu+4)
            endif
         if ((_LINUX_)) then
            b(4) = buff(nbu+1)
            b(3) = buff(nbu+2)
            b(2) = buff(nbu+3)
            b(1) = buff(nbu+4)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = ii
         enddo
 
      return
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/WFC3IR/read_wfc3ir_flt_IMPROVED.f"
c****                                         
c*********************************************
                                              
 
c
c reads in a wfc3/ir image
c
      subroutine read_wfc3ir_flt(FILENAME,pixr)
      implicit none
 
      real      pixr(1014,1014)
 
      real      pix(1014,1014), pmax
      integer*2 pox(1014,1014)
      integer   pux(1014,1014)
      integer  i, j
      integer ii,jj
      real xsum, ysum, psum, msky
      real xbar, ybar, mag
      real mbar_sky
      character*80 FILENAME
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      real EXPTIME
      integer k
      real pval
      integer ival
 
      real      pixi(1014,1014)
      real      pixh(1014,1014)
      integer*2 pixo(1014,1014)
      integer*2 pixi2(1014,1014)
 
 
      real rsum
      real dd
      integer Ls
      real    plist(9)
      integer NIT, NFIXN, NFIXT
      byte    pixi2b(1014,1014,16), b
      integer*2 i2
 
      do i = 0001, 1014
      do j = 0001, 1014
         pixi(i,j)  = 0.
         pixi2(i,j) = 0
         enddo
         enddo
 
      call readfits_r4e(FILENAME,pixi ,1014,1014,1)
      call readfits_i2e(FILENAME,pixi2,1014,1014,3)
 
      do i = 0001, 1014
      do j = 0001, 1014
         i2 = pixi2(i,j)
         pixi2b(i,j,16) = i2/32768
         i2 = i2-i2/32768*32768
         pixi2b(i,j,15) = i2/16384
         i2 = i2-i2/16384*16384
         pixi2b(i,j,14) = i2/08192
         i2 = i2-i2/08192*08192
         pixi2b(i,j,13) = i2/04096
         i2 = i2-i2/04096*04096
         pixi2b(i,j,12) = i2/02048
         i2 = i2-i2/02048*02048
         pixi2b(i,j,11) = i2/01024
         i2 = i2-i2/01024*01024
         pixi2b(i,j,10) = i2/00512
         i2 = i2-i2/00512*00512
         pixi2b(i,j,09) = i2/00256
         i2 = i2-i2/00256*00256
         pixi2b(i,j,08) = i2/00128
         i2 = i2-i2/00128*00128
         pixi2b(i,j,07) = i2/00064
         i2 = i2-i2/00064*00064
         pixi2b(i,j,06) = i2/00032
         i2 = i2-i2/00032*00032
         pixi2b(i,j,05) = i2/00016
         i2 = i2-i2/00016*00016
         pixi2b(i,j,04) = i2/00008
         i2 = i2-i2/00008*00008
         pixi2b(i,j,03) = i2/00004
         i2 = i2-i2/00004*00004
         pixi2b(i,j,02) = i2/00002
         i2 = i2-i2/00002*00002
         pixi2b(i,j,01) = i2
         if (pixi2b(i,j,03).eq.1) pixi(i,j) = -9999
         if (pixi2b(i,j,09).eq.1) pixi(i,j) = -9999
         pixr(i,j) = pixi(i,j)
         enddo
         enddo
 
 
      do i = 1, 1014
      do j = 1, 1014
         pixh(i,j) = pixr(i,j)
         enddo
         enddo
 
      do i = 0001, 1014
      do j = 0001, 1014
         rsum = sqrt((i-359.)**2+(j-55.)**2)
         if (rsum.lt.25) pixr(i,j) = -9999
         enddo
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_r4e.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------------------
c
c reads an r4 fits image, with extensions (reads in one extension)
c
 
      subroutine readfits_r4e(FILE,pix,NDIMX,NDIMY,NEXTENU)
      implicit none
 
      character*(*) FILE
      integer NDIMX,NDIMY
      real    pix(NDIMX,NDIMY)
      integer NEXTENU
 
      character*199 FILEU
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      integer naxes
      integer laxis(3)
      common/laxis3_/laxis
 
      integer NXF
      integer NYF
 
      character*8  field
      character*40 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer j
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      character*70 HDR(25)
      common/HDR/HDR
 
      logical DIAG
      data DIAG /.false./
      integer NEND
      integer iend
 
      if (DIAG) then
         print*,'enter readfits_r4e...... '
         print*,'FILEi: ',FILE(1:60)
         endif
 
      FILEU = FILE
      iend = 0
      do i = 196,1,-1
         if (DIAG) print*,i,iend,FILE(i:i+4)
         if (FILE(i:i+4).eq.'.fits') iend = i+4
         enddo
      if (DIAG) then
         print*,'iend: ',iend
         endif
      if (iend.eq.0) stop 'NO .fits in FILENAME'
      FILEU = FILE(1:iend)
 
      if (DIAG) then
         print*,'purge...'
         print*,'FILEu: ',FILEU
         endif
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened ',FILEU
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
 
      BSCALE = 1.0
      BZERO  = 0.0
 
      NEND = 0
      i = 0
      nread = 0
 100  continue
      i = i + 1
      if (DIAG) print*,'READREC: ',i
      read(10,rec=i,iostat=ios) buffc
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+51)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(1) = stream
         if (field.eq.'FILTER  ') INFO(2) = stream
         if (field.eq.'FILTNAM1') INFO(2) = stream
         if (field.eq.'FILENAME') INFO(3) = stream
         if (field.eq.'DATE-OBS') INFO(4) = stream
         if (field.eq.'TIME-OBS') INFO(5) = stream
         if (field.eq.'DEC_TARG') INFO(6) = stream
         if (field.eq.'RA_TARG ') INFO(7) = stream
         if (field.eq.'PA_V3   ') INFO(8) = stream
         if (field.eq.'PROPOSID') INFO(9) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER ')  HDR(22) = stream
         if (field.eq.'FILTER1 ') HDR(22) = stream
         if (field.eq.'FILTER2 ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
 
         if (field.eq.'END     ') then
           !print*,' ---> NEND: ',NEND,NEXTENU
            if (NEND.eq.NEXTENU) goto 101
            NEND = NEND + 1
            endif
         enddo
      goto 100
 101  continue
 
      nread = nread + 1
      if (DIAG) then
         print*,' '
         print*,'----------------------------------------'
         print*,'  NREAD: ',nread
         print*,'NEXTEND: ',nextend
         print*,'  NAXIS: ',naxes
         print*,'  LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,' BITPIX: ',bitpix
         print*,' BSCALE: ',bscale
         print*,'  BZERO: ',bzero
         print*,'  NDIMX: ',NDIMX
         print*,'  NDIMY: ',NDIMY
         print*,' '
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      NXF = laxis(1)
      NYF = laxis(2)
 
      nbper  = 4*laxis(1)*laxis(2)
      npt    = laxis(1)*laxis(2)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      if (BITPIX.ne.-32) then
         print*,'readfits_r4e...'
         print*,'BITPIX: ',BITPIX
         print*,'prob'
         stop
         endif
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
c        call buff2pix_r4(buffb,pix,np1,npt)
         call buff2pix_r4_edge(buffb,pix,np1,npt,
     .                         NDIMX,NDIMY,laxis(1),laxis(2))
         if (DIAG) write(*,1115) i,np1,np2,npt
 1115    format(1x,i8,1x,i10,1x,i10,1x,i10)
         enddo
 
      close(10)
      if (DIAG) print*,'...closed ',FILEU
      return
 
 900  continue
      print*,'READFITS ERROR'
      stop
 
      end
 
 
c------------------------------------------------------
c
c
      subroutine buff2pix_r4_edge(buff,pix,n1,nt,
     .                            NXP,NYP,NXF,NYF)
      implicit none
c     character buff(2880)
      byte buff(2880)
      integer NXP,NYP
      real pix(NXP,NYP)
      integer n1,nt
      integer NXF,NYF
 
      real pbuff(720)
      common /sneaky/pbuff
      integer i
      integer npu, nbu
      integer NX, NY
 
      byte b(4)
      real r
      equivalence(r,b)
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (npu.ge.1.and.npu.le.nt) then
            NX = npu - (npu-1)/NXF*NXF
            NY = 1 + (npu-1)/NXF
            if (NX.le.NXP.and.NY.le.NYP) then
               if (.not.(_LINUX_)) then
                  b(1) = buff(nbu+1)
                  b(2) = buff(nbu+2)
                  b(3) = buff(nbu+3)
                  b(4) = buff(nbu+4)
                  endif
               if ((_LINUX_)) then
                  b(4) = buff(nbu+1)
                  b(3) = buff(nbu+2)
                  b(2) = buff(nbu+3)
                  b(1) = buff(nbu+4)
                  endif
               if (npu.ge.1.and.npu.le.nt) pix(NX,NY) = r
               endif
            endif
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_i2e.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------------
c
c read in an i2 image with extensions
c
      subroutine readfits_i2e(FILEI,pix,NDIMX,NDIMY,NEXTENU)
      implicit none
 
      character*(*) FILEI
      integer   NDIMX,NDIMY
      integer   NEXTENU
      integer*2 pix(NDIMX,NDIMY)
 
      character*199 FILEU
      character*070 INFO(10)
      common / fitsinfo / INFO
 
      integer naxes
      integer laxis(3)
      integer NXF
      integer NYF
 
      character*8  field
      character*40 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer j
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      character*70 HDR(25)
      common/HDR/HDR
 
      logical DIAG
      data DIAG /.false./
      integer NEND
      integer ii
 
      FILEU = FILEI
      do i = 195,2,-1
         if (FILEI(i:i+4).eq.'.fits') then
            FILEU = FILEI(1:i+4)
            do ii = i+5, 80
               FILEU(ii:ii) = ' '
               enddo
            endif
         enddo
 
      if (DIAG) then
         print*,'enter readfits_i2e...'
         write(*,'(''FILEI: '',80a)') FILEI
         write(*,'(''FILEU: '',80a)') FILEU
         endif
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened'
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
 
      BSCALE = 1.0
      BZERO  = 0.0
 
      NEXTEND = 0
      NEND = -1
      i = 0
      nread = 0
 100  continue
      i = i + 1
      if (DIAG) print*,'READREC: ',i
      read(10,rec=i,iostat=ios) buffc
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+51)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(1) = stream
         if (field.eq.'FILTNAM1') INFO(2) = stream
         if (field.eq.'FILENAME') INFO(3) = stream
         if (field.eq.'DATE-OBS') INFO(4) = stream
         if (field.eq.'TIME-OBS') INFO(5) = stream
         if (field.eq.'DEC_TARG') INFO(6) = stream
         if (field.eq.'RA_TARG ') INFO(7) = stream
         if (field.eq.'DEC_DEG ') INFO(6) = stream
         if (field.eq.'RA_DEG  ') INFO(7) = stream
         if (field.eq.'PA_V3   ') INFO(8) = stream
         if (field.eq.'PROPOSID') INFO(9) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'RA_DEG  ') HDR(19) = stream
         if (field.eq.'DEC_DEG ') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER1 ') HDR(22) = stream
         if (field.eq.'FILTER2 ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
         if (field.eq.'CCDGAIN ') HDR(25) = stream
         if (field.eq.'END     ') then
            if (NEXTENU.gt.NEXTEND.and.NEXTENU.gt.1) then
               print*,'                     '
               write(*,'(''readfits_i2e: '',80a)') FILEI
               print*,'     NEXTENU.lt.NEXTEND...'
               print*,'     ---> NEXTEND: ',NEXTEND
               print*,'     ---> NEXTENU: ',NEXTENU
               print*,'                          '
               stop
               endif
            NEND = NEND + 1
            if (NEND.ge.1) goto 101
            endif
         enddo
      goto 100
 101  continue
 
      nread = nread + 1
      if (DIAG) then
         print*,' '
         print*,'----------------------------------------'
         print*,'   NREAD: ',nread
         print*,' NEXTEND: ',nextend
         print*,' NEXTENU: ',nextenu
         print*,'   NAXIS: ',naxes
         print*,'   LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,'  BITPIX: ',bitpix
         print*,'  BSCALE: ',bscale
         print*,'   BZERO: ',bzero
         print*,'   NDIMX: ',NDIMX
         print*,'   NDIMY: ',NDIMY
         print*,' '
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      NXF = laxis(1)
      NYF = laxis(2)
 
      nbper  = 2*laxis(1)*laxis(2)
      npt    =   laxis(1)*laxis(2)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
c     print*,'NEND: ',NEND,NEXTENU,BITPIX
      if (NEND.ne.NEXTENU) goto 100
 
      if (BITPIX.ne.16) then
         print*,'BITPIX: ',BITPIX
         print*,'prob'
         stop
         endif
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/2 + 1
         np2 = (nbyteE-nbyte1)/2 + 1
         call buff2pix_i2e(buffb,pix(1,1),np1,npt)
         if (DIAG) write(*,1115) i,np1,np2,npt,i/laxis(1)
 1115    format(1x,i8,1x,i10,1x,i10,1x,i10,1x,i6.6)
         enddo
 
 
      close(10)
      return
 
 900  continue
      print*,'READFITS_I2E ERROR'
      print*,'FILEU: ',FILEU
      stop
 
      end
 
 
 
 
      subroutine buff2pix_i2e(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      integer*2 pix(*)
      integer n1,nt
 
      byte b(2)
      integer ii
      equivalence(ii,b)
 
      integer i, npu, nbu
 
      do i = 1, 1440
         npu = n1+i-1
         nbu = (i-1)*2
         if (.not.(_LINUX_)) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            endif
         if ((_LINUX_)) then
            b(2) = buff(nbu+1)
            b(1) = buff(nbu+2)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = ii
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/GC/WFC3IR/wfc3ir_zpa.f"
c****                                         
c*********************************************
                                              
 
c----------------------------------------
c
c
c
      real function wfc3ir_zpa(xr,yr)
      implicit none
 
      real*8 xr, yr
 
      real    ri,rj
      integer ii,jj
      real    fi,fj
 
      real wfc3ir_zp_arr(64,64)
      common /wfc3ir_zp_arr_/wfc3ir_zp_arr
 
      ri = 1 + (xr-8)/16.0
      rj = 1 + (yr-8)/16.0
      if (ri.le. 1.0) ri =  1.00
      if (rj.le. 1.0) rj =  1.00
      if (ri.ge.64.0) ri = 63.99
      if (rj.ge.64.0) rj = 63.99
      ii = int(ri)
      jj = int(rj)
      fi = ri-ii
      fj = rj-jj
 
      wfc3ir_zpa = (1-fi)*(1-fj)*wfc3ir_zp_arr(ii  ,jj  )
     .           + (1-fi)*( fj )*wfc3ir_zp_arr(ii  ,jj+1)
     .           + ( fi )*(1-fj)*wfc3ir_zp_arr(ii+1,jj  )
     .           + ( fi )*( fj )*wfc3ir_zp_arr(ii+1,jj+1)
 
      return
      end
 
 
 
c---------------------------------------------------------------------
c
c
c
      block data wfc3ir_zp_arr_BD
      real wfc3ir_zp_arr(64,64)
      common /wfc3ir_zp_arr_/wfc3ir_zp_arr
      data wfc3ir_zp_arr/
     . 0.0469, 0.0469, 0.0469, 0.0469, 0.0469, 0.0469, 0.0469, 0.0470,
     . 0.0470, 0.0470, 0.0470, 0.0470, 0.0471, 0.0471, 0.0471, 0.0471,
     . 0.0472, 0.0472, 0.0472, 0.0472, 0.0472, 0.0472, 0.0472, 0.0473,
     . 0.0473, 0.0474, 0.0475, 0.0475, 0.0475, 0.0475, 0.0475, 0.0476,
     . 0.0476, 0.0475, 0.0475, 0.0475, 0.0475, 0.0475, 0.0474, 0.0474,
     . 0.0474, 0.0474, 0.0474, 0.0474, 0.0475, 0.0475, 0.0475, 0.0476,
     . 0.0476, 0.0476, 0.0477, 0.0477, 0.0477, 0.0478, 0.0478, 0.0478,
     . 0.0479, 0.0479, 0.0479, 0.0479, 0.0479, 0.0480, 0.0480, 0.0480,
     . 0.0453, 0.0454, 0.0454, 0.0454, 0.0454, 0.0454, 0.0454, 0.0454,
     . 0.0454, 0.0455, 0.0455, 0.0455, 0.0455, 0.0456, 0.0456, 0.0456,
     . 0.0456, 0.0456, 0.0457, 0.0456, 0.0456, 0.0457, 0.0457, 0.0457,
     . 0.0458, 0.0459, 0.0459, 0.0459, 0.0460, 0.0460, 0.0460, 0.0460,
     . 0.0460, 0.0460, 0.0460, 0.0460, 0.0460, 0.0459, 0.0459, 0.0459,
     . 0.0459, 0.0459, 0.0459, 0.0459, 0.0460, 0.0460, 0.0460, 0.0460,
     . 0.0460, 0.0461, 0.0462, 0.0462, 0.0462, 0.0462, 0.0462, 0.0462,
     . 0.0463, 0.0464, 0.0464, 0.0464, 0.0464, 0.0464, 0.0464, 0.0465,
     . 0.0438, 0.0438, 0.0438, 0.0439, 0.0439, 0.0439, 0.0439, 0.0439,
     . 0.0439, 0.0439, 0.0440, 0.0440, 0.0440, 0.0440, 0.0440, 0.0441,
     . 0.0441, 0.0441, 0.0441, 0.0441, 0.0441, 0.0441, 0.0442, 0.0442,
     . 0.0442, 0.0444, 0.0444, 0.0444, 0.0444, 0.0445, 0.0445, 0.0445,
     . 0.0445, 0.0445, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0443,
     . 0.0443, 0.0443, 0.0443, 0.0443, 0.0444, 0.0444, 0.0445, 0.0445,
     . 0.0445, 0.0445, 0.0446, 0.0447, 0.0447, 0.0447, 0.0447, 0.0447,
     . 0.0448, 0.0448, 0.0448, 0.0449, 0.0449, 0.0449, 0.0449, 0.0449,
     . 0.0423, 0.0423, 0.0423, 0.0423, 0.0423, 0.0424, 0.0424, 0.0424,
     . 0.0424, 0.0424, 0.0424, 0.0424, 0.0425, 0.0425, 0.0425, 0.0425,
     . 0.0426, 0.0426, 0.0426, 0.0425, 0.0426, 0.0426, 0.0426, 0.0427,
     . 0.0427, 0.0428, 0.0429, 0.0429, 0.0429, 0.0429, 0.0430, 0.0429,
     . 0.0429, 0.0429, 0.0429, 0.0429, 0.0429, 0.0429, 0.0428, 0.0428,
     . 0.0428, 0.0428, 0.0428, 0.0428, 0.0429, 0.0429, 0.0429, 0.0430,
     . 0.0430, 0.0430, 0.0431, 0.0431, 0.0431, 0.0432, 0.0432, 0.0432,
     . 0.0433, 0.0433, 0.0433, 0.0433, 0.0433, 0.0434, 0.0434, 0.0434,
     . 0.0408, 0.0408, 0.0408, 0.0408, 0.0408, 0.0408, 0.0408, 0.0409,
     . 0.0409, 0.0409, 0.0409, 0.0409, 0.0409, 0.0410, 0.0410, 0.0410,
     . 0.0410, 0.0410, 0.0411, 0.0410, 0.0410, 0.0411, 0.0411, 0.0411,
     . 0.0412, 0.0413, 0.0413, 0.0414, 0.0414, 0.0414, 0.0414, 0.0414,
     . 0.0414, 0.0414, 0.0414, 0.0414, 0.0413, 0.0413, 0.0413, 0.0413,
     . 0.0413, 0.0413, 0.0412, 0.0412, 0.0414, 0.0414, 0.0414, 0.0414,
     . 0.0415, 0.0415, 0.0416, 0.0416, 0.0416, 0.0416, 0.0416, 0.0416,
     . 0.0417, 0.0418, 0.0418, 0.0418, 0.0418, 0.0418, 0.0418, 0.0419,
     . 0.0392, 0.0393, 0.0393, 0.0393, 0.0393, 0.0393, 0.0393, 0.0393,
     . 0.0393, 0.0394, 0.0394, 0.0394, 0.0394, 0.0394, 0.0394, 0.0395,
     . 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0396, 0.0396,
     . 0.0396, 0.0398, 0.0398, 0.0398, 0.0399, 0.0399, 0.0399, 0.0399,
     . 0.0398, 0.0398, 0.0398, 0.0398, 0.0398, 0.0398, 0.0397, 0.0397,
     . 0.0397, 0.0397, 0.0397, 0.0397, 0.0398, 0.0399, 0.0399, 0.0399,
     . 0.0399, 0.0400, 0.0401, 0.0401, 0.0401, 0.0401, 0.0401, 0.0401,
     . 0.0402, 0.0402, 0.0403, 0.0403, 0.0403, 0.0403, 0.0403, 0.0403,
     . 0.0378, 0.0378, 0.0378, 0.0378, 0.0378, 0.0378, 0.0378, 0.0378,
     . 0.0379, 0.0379, 0.0379, 0.0379, 0.0379, 0.0379, 0.0379, 0.0380,
     . 0.0380, 0.0380, 0.0380, 0.0380, 0.0380, 0.0380, 0.0380, 0.0380,
     . 0.0381, 0.0382, 0.0382, 0.0383, 0.0383, 0.0383, 0.0383, 0.0382,
     . 0.0382, 0.0382, 0.0382, 0.0382, 0.0382, 0.0382, 0.0382, 0.0382,
     . 0.0382, 0.0382, 0.0382, 0.0382, 0.0384, 0.0384, 0.0384, 0.0384,
     . 0.0384, 0.0384, 0.0385, 0.0385, 0.0385, 0.0386, 0.0386, 0.0386,
     . 0.0387, 0.0387, 0.0387, 0.0387, 0.0387, 0.0388, 0.0388, 0.0388,
     . 0.0362, 0.0363, 0.0363, 0.0363, 0.0363, 0.0363, 0.0363, 0.0363,
     . 0.0363, 0.0364, 0.0364, 0.0364, 0.0364, 0.0364, 0.0364, 0.0364,
     . 0.0365, 0.0365, 0.0365, 0.0364, 0.0365, 0.0365, 0.0365, 0.0365,
     . 0.0365, 0.0367, 0.0367, 0.0367, 0.0367, 0.0368, 0.0368, 0.0367,
     . 0.0367, 0.0367, 0.0367, 0.0367, 0.0367, 0.0367, 0.0367, 0.0367,
     . 0.0367, 0.0367, 0.0367, 0.0367, 0.0368, 0.0369, 0.0369, 0.0369,
     . 0.0369, 0.0369, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0371,
     . 0.0372, 0.0372, 0.0372, 0.0372, 0.0372, 0.0372, 0.0373, 0.0373,
     . 0.0347, 0.0347, 0.0347, 0.0348, 0.0348, 0.0348, 0.0348, 0.0348,
     . 0.0348, 0.0348, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349,
     . 0.0349, 0.0349, 0.0350, 0.0349, 0.0349, 0.0350, 0.0350, 0.0350,
     . 0.0350, 0.0352, 0.0352, 0.0352, 0.0352, 0.0352, 0.0353, 0.0352,
     . 0.0352, 0.0352, 0.0352, 0.0352, 0.0352, 0.0352, 0.0352, 0.0352,
     . 0.0352, 0.0352, 0.0352, 0.0352, 0.0353, 0.0353, 0.0353, 0.0354,
     . 0.0354, 0.0354, 0.0355, 0.0355, 0.0355, 0.0355, 0.0355, 0.0355,
     . 0.0356, 0.0357, 0.0357, 0.0357, 0.0357, 0.0357, 0.0357, 0.0357,
     . 0.0332, 0.0332, 0.0332, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333,
     . 0.0333, 0.0333, 0.0333, 0.0334, 0.0334, 0.0334, 0.0334, 0.0334,
     . 0.0334, 0.0334, 0.0334, 0.0334, 0.0334, 0.0334, 0.0335, 0.0335,
     . 0.0335, 0.0337, 0.0337, 0.0337, 0.0337, 0.0337, 0.0337, 0.0336,
     . 0.0336, 0.0336, 0.0336, 0.0336, 0.0337, 0.0336, 0.0336, 0.0336,
     . 0.0336, 0.0336, 0.0336, 0.0337, 0.0338, 0.0338, 0.0338, 0.0338,
     . 0.0338, 0.0339, 0.0340, 0.0340, 0.0340, 0.0340, 0.0340, 0.0340,
     . 0.0341, 0.0341, 0.0341, 0.0342, 0.0342, 0.0342, 0.0342, 0.0342,
     . 0.0317, 0.0317, 0.0317, 0.0317, 0.0317, 0.0318, 0.0318, 0.0318,
     . 0.0318, 0.0318, 0.0318, 0.0318, 0.0318, 0.0318, 0.0319, 0.0319,
     . 0.0319, 0.0319, 0.0319, 0.0319, 0.0319, 0.0319, 0.0319, 0.0320,
     . 0.0320, 0.0321, 0.0321, 0.0322, 0.0322, 0.0322, 0.0322, 0.0321,
     . 0.0321, 0.0321, 0.0321, 0.0321, 0.0321, 0.0321, 0.0321, 0.0321,
     . 0.0321, 0.0321, 0.0321, 0.0321, 0.0323, 0.0323, 0.0323, 0.0323,
     . 0.0323, 0.0323, 0.0324, 0.0324, 0.0324, 0.0325, 0.0325, 0.0325,
     . 0.0326, 0.0326, 0.0326, 0.0326, 0.0326, 0.0327, 0.0327, 0.0327,
     . 0.0302, 0.0302, 0.0302, 0.0302, 0.0302, 0.0303, 0.0302, 0.0303,
     . 0.0303, 0.0303, 0.0303, 0.0303, 0.0303, 0.0303, 0.0303, 0.0304,
     . 0.0304, 0.0304, 0.0304, 0.0304, 0.0304, 0.0304, 0.0304, 0.0304,
     . 0.0305, 0.0306, 0.0306, 0.0306, 0.0307, 0.0307, 0.0307, 0.0306,
     . 0.0306, 0.0306, 0.0306, 0.0306, 0.0306, 0.0306, 0.0306, 0.0306,
     . 0.0306, 0.0306, 0.0306, 0.0306, 0.0308, 0.0308, 0.0308, 0.0308,
     . 0.0308, 0.0308, 0.0309, 0.0309, 0.0309, 0.0309, 0.0310, 0.0310,
     . 0.0311, 0.0311, 0.0311, 0.0311, 0.0311, 0.0311, 0.0311, 0.0312,
     . 0.0287, 0.0287, 0.0287, 0.0287, 0.0287, 0.0288, 0.0287, 0.0288,
     . 0.0288, 0.0288, 0.0288, 0.0288, 0.0288, 0.0288, 0.0288, 0.0288,
     . 0.0288, 0.0288, 0.0288, 0.0288, 0.0288, 0.0288, 0.0289, 0.0289,
     . 0.0289, 0.0291, 0.0291, 0.0291, 0.0291, 0.0291, 0.0291, 0.0290,
     . 0.0290, 0.0290, 0.0290, 0.0290, 0.0290, 0.0290, 0.0291, 0.0291,
     . 0.0291, 0.0291, 0.0291, 0.0291, 0.0292, 0.0292, 0.0292, 0.0292,
     . 0.0292, 0.0292, 0.0293, 0.0294, 0.0294, 0.0294, 0.0294, 0.0294,
     . 0.0295, 0.0296, 0.0296, 0.0296, 0.0296, 0.0297, 0.0297, 0.0297,
     . 0.0272, 0.0273, 0.0273, 0.0273, 0.0273, 0.0273, 0.0273, 0.0273,
     . 0.0273, 0.0273, 0.0273, 0.0273, 0.0273, 0.0273, 0.0273, 0.0273,
     . 0.0273, 0.0273, 0.0273, 0.0273, 0.0273, 0.0273, 0.0273, 0.0274,
     . 0.0274, 0.0275, 0.0275, 0.0275, 0.0276, 0.0276, 0.0276, 0.0275,
     . 0.0275, 0.0275, 0.0275, 0.0275, 0.0275, 0.0275, 0.0275, 0.0275,
     . 0.0276, 0.0276, 0.0276, 0.0276, 0.0277, 0.0277, 0.0277, 0.0277,
     . 0.0277, 0.0277, 0.0278, 0.0278, 0.0278, 0.0278, 0.0279, 0.0279,
     . 0.0280, 0.0280, 0.0281, 0.0281, 0.0281, 0.0282, 0.0282, 0.0282,
     . 0.0257, 0.0257, 0.0257, 0.0258, 0.0258, 0.0258, 0.0257, 0.0258,
     . 0.0258, 0.0258, 0.0258, 0.0258, 0.0258, 0.0258, 0.0258, 0.0258,
     . 0.0258, 0.0258, 0.0258, 0.0257, 0.0258, 0.0258, 0.0258, 0.0259,
     . 0.0259, 0.0260, 0.0260, 0.0260, 0.0260, 0.0260, 0.0260, 0.0260,
     . 0.0259, 0.0260, 0.0260, 0.0260, 0.0260, 0.0260, 0.0260, 0.0260,
     . 0.0260, 0.0261, 0.0261, 0.0261, 0.0262, 0.0262, 0.0262, 0.0262,
     . 0.0262, 0.0262, 0.0263, 0.0263, 0.0263, 0.0263, 0.0263, 0.0264,
     . 0.0265, 0.0265, 0.0266, 0.0266, 0.0266, 0.0267, 0.0267, 0.0267,
     . 0.0242, 0.0242, 0.0243, 0.0243, 0.0243, 0.0243, 0.0242, 0.0243,
     . 0.0243, 0.0243, 0.0243, 0.0243, 0.0243, 0.0243, 0.0243, 0.0243,
     . 0.0243, 0.0243, 0.0243, 0.0242, 0.0243, 0.0243, 0.0243, 0.0243,
     . 0.0244, 0.0245, 0.0245, 0.0245, 0.0245, 0.0245, 0.0245, 0.0244,
     . 0.0244, 0.0244, 0.0244, 0.0244, 0.0245, 0.0245, 0.0245, 0.0245,
     . 0.0245, 0.0246, 0.0246, 0.0246, 0.0247, 0.0247, 0.0246, 0.0246,
     . 0.0246, 0.0246, 0.0247, 0.0248, 0.0248, 0.0248, 0.0248, 0.0249,
     . 0.0250, 0.0250, 0.0250, 0.0251, 0.0251, 0.0251, 0.0252, 0.0252,
     . 0.0227, 0.0227, 0.0227, 0.0228, 0.0228, 0.0228, 0.0227, 0.0228,
     . 0.0228, 0.0228, 0.0228, 0.0228, 0.0228, 0.0228, 0.0228, 0.0228,
     . 0.0228, 0.0228, 0.0228, 0.0227, 0.0228, 0.0228, 0.0228, 0.0228,
     . 0.0229, 0.0229, 0.0229, 0.0229, 0.0230, 0.0230, 0.0230, 0.0229,
     . 0.0229, 0.0229, 0.0229, 0.0229, 0.0229, 0.0230, 0.0230, 0.0230,
     . 0.0230, 0.0231, 0.0231, 0.0231, 0.0231, 0.0231, 0.0231, 0.0231,
     . 0.0231, 0.0231, 0.0232, 0.0233, 0.0233, 0.0233, 0.0233, 0.0233,
     . 0.0235, 0.0235, 0.0235, 0.0236, 0.0236, 0.0236, 0.0237, 0.0237,
     . 0.0212, 0.0212, 0.0212, 0.0213, 0.0213, 0.0213, 0.0212, 0.0213,
     . 0.0213, 0.0213, 0.0213, 0.0213, 0.0213, 0.0213, 0.0213, 0.0213,
     . 0.0213, 0.0213, 0.0213, 0.0212, 0.0212, 0.0213, 0.0213, 0.0213,
     . 0.0214, 0.0214, 0.0214, 0.0214, 0.0214, 0.0214, 0.0214, 0.0214,
     . 0.0214, 0.0214, 0.0214, 0.0214, 0.0214, 0.0215, 0.0215, 0.0215,
     . 0.0215, 0.0216, 0.0216, 0.0216, 0.0216, 0.0216, 0.0216, 0.0216,
     . 0.0216, 0.0216, 0.0217, 0.0217, 0.0218, 0.0218, 0.0218, 0.0218,
     . 0.0219, 0.0220, 0.0220, 0.0220, 0.0221, 0.0221, 0.0221, 0.0222,
     . 0.0198, 0.0198, 0.0198, 0.0198, 0.0198, 0.0198, 0.0198, 0.0198,
     . 0.0198, 0.0198, 0.0198, 0.0198, 0.0198, 0.0198, 0.0198, 0.0198,
     . 0.0198, 0.0198, 0.0198, 0.0197, 0.0198, 0.0198, 0.0198, 0.0198,
     . 0.0199, 0.0199, 0.0199, 0.0199, 0.0199, 0.0199, 0.0199, 0.0199,
     . 0.0199, 0.0199, 0.0199, 0.0199, 0.0199, 0.0200, 0.0200, 0.0201,
     . 0.0201, 0.0201, 0.0201, 0.0201, 0.0201, 0.0201, 0.0201, 0.0201,
     . 0.0201, 0.0201, 0.0202, 0.0203, 0.0203, 0.0203, 0.0203, 0.0203,
     . 0.0205, 0.0205, 0.0205, 0.0206, 0.0206, 0.0206, 0.0207, 0.0207,
     . 0.0184, 0.0184, 0.0184, 0.0184, 0.0184, 0.0184, 0.0184, 0.0184,
     . 0.0184, 0.0184, 0.0184, 0.0184, 0.0184, 0.0184, 0.0184, 0.0184,
     . 0.0184, 0.0185, 0.0185, 0.0184, 0.0184, 0.0184, 0.0184, 0.0184,
     . 0.0184, 0.0184, 0.0184, 0.0184, 0.0184, 0.0184, 0.0184, 0.0184,
     . 0.0184, 0.0184, 0.0185, 0.0185, 0.0185, 0.0186, 0.0187, 0.0187,
     . 0.0187, 0.0187, 0.0188, 0.0188, 0.0188, 0.0188, 0.0188, 0.0188,
     . 0.0188, 0.0189, 0.0190, 0.0190, 0.0190, 0.0190, 0.0191, 0.0191,
     . 0.0192, 0.0192, 0.0193, 0.0193, 0.0193, 0.0193, 0.0194, 0.0194,
     . 0.0169, 0.0169, 0.0169, 0.0169, 0.0169, 0.0169, 0.0169, 0.0169,
     . 0.0169, 0.0169, 0.0169, 0.0169, 0.0169, 0.0169, 0.0169, 0.0169,
     . 0.0170, 0.0170, 0.0170, 0.0169, 0.0169, 0.0169, 0.0169, 0.0169,
     . 0.0169, 0.0168, 0.0168, 0.0169, 0.0169, 0.0169, 0.0169, 0.0169,
     . 0.0169, 0.0169, 0.0169, 0.0170, 0.0170, 0.0171, 0.0172, 0.0172,
     . 0.0172, 0.0172, 0.0173, 0.0173, 0.0173, 0.0173, 0.0173, 0.0174,
     . 0.0174, 0.0174, 0.0175, 0.0175, 0.0175, 0.0175, 0.0175, 0.0176,
     . 0.0177, 0.0177, 0.0177, 0.0178, 0.0178, 0.0178, 0.0178, 0.0179,
     . 0.0154, 0.0154, 0.0154, 0.0154, 0.0154, 0.0154, 0.0154, 0.0154,
     . 0.0154, 0.0154, 0.0154, 0.0154, 0.0154, 0.0154, 0.0154, 0.0155,
     . 0.0155, 0.0155, 0.0155, 0.0154, 0.0154, 0.0154, 0.0154, 0.0154,
     . 0.0154, 0.0153, 0.0153, 0.0153, 0.0153, 0.0154, 0.0154, 0.0153,
     . 0.0154, 0.0154, 0.0154, 0.0155, 0.0155, 0.0156, 0.0157, 0.0157,
     . 0.0157, 0.0157, 0.0158, 0.0158, 0.0158, 0.0159, 0.0159, 0.0159,
     . 0.0159, 0.0159, 0.0160, 0.0160, 0.0160, 0.0160, 0.0160, 0.0161,
     . 0.0161, 0.0162, 0.0162, 0.0162, 0.0162, 0.0163, 0.0163, 0.0163,
     . 0.0139, 0.0139, 0.0139, 0.0139, 0.0139, 0.0139, 0.0139, 0.0139,
     . 0.0139, 0.0139, 0.0139, 0.0139, 0.0139, 0.0140, 0.0140, 0.0140,
     . 0.0140, 0.0140, 0.0140, 0.0139, 0.0139, 0.0139, 0.0139, 0.0139,
     . 0.0139, 0.0138, 0.0138, 0.0138, 0.0138, 0.0138, 0.0138, 0.0138,
     . 0.0139, 0.0139, 0.0139, 0.0139, 0.0140, 0.0141, 0.0142, 0.0142,
     . 0.0142, 0.0142, 0.0143, 0.0143, 0.0144, 0.0144, 0.0144, 0.0144,
     . 0.0144, 0.0144, 0.0144, 0.0145, 0.0145, 0.0145, 0.0145, 0.0145,
     . 0.0146, 0.0146, 0.0147, 0.0147, 0.0147, 0.0147, 0.0148, 0.0148,
     . 0.0124, 0.0124, 0.0124, 0.0124, 0.0123, 0.0123, 0.0124, 0.0124,
     . 0.0124, 0.0124, 0.0124, 0.0124, 0.0124, 0.0125, 0.0125, 0.0125,
     . 0.0125, 0.0125, 0.0125, 0.0124, 0.0124, 0.0124, 0.0124, 0.0124,
     . 0.0124, 0.0123, 0.0123, 0.0123, 0.0123, 0.0123, 0.0123, 0.0123,
     . 0.0124, 0.0124, 0.0124, 0.0124, 0.0125, 0.0126, 0.0127, 0.0127,
     . 0.0127, 0.0128, 0.0128, 0.0128, 0.0129, 0.0129, 0.0129, 0.0129,
     . 0.0129, 0.0129, 0.0129, 0.0129, 0.0130, 0.0130, 0.0130, 0.0130,
     . 0.0131, 0.0131, 0.0131, 0.0132, 0.0132, 0.0132, 0.0132, 0.0133,
     . 0.0109, 0.0109, 0.0109, 0.0108, 0.0108, 0.0108, 0.0109, 0.0109,
     . 0.0109, 0.0109, 0.0109, 0.0109, 0.0109, 0.0110, 0.0110, 0.0110,
     . 0.0110, 0.0110, 0.0110, 0.0109, 0.0109, 0.0109, 0.0109, 0.0109,
     . 0.0109, 0.0108, 0.0108, 0.0108, 0.0108, 0.0108, 0.0108, 0.0108,
     . 0.0108, 0.0109, 0.0109, 0.0109, 0.0110, 0.0111, 0.0112, 0.0112,
     . 0.0112, 0.0113, 0.0113, 0.0113, 0.0114, 0.0114, 0.0114, 0.0114,
     . 0.0114, 0.0114, 0.0114, 0.0114, 0.0114, 0.0115, 0.0115, 0.0115,
     . 0.0116, 0.0116, 0.0116, 0.0116, 0.0117, 0.0117, 0.0117, 0.0117,
     . 0.0096, 0.0096, 0.0096, 0.0096, 0.0096, 0.0095, 0.0096, 0.0096,
     . 0.0095, 0.0095, 0.0095, 0.0095, 0.0096, 0.0096, 0.0096, 0.0097,
     . 0.0097, 0.0097, 0.0096, 0.0095, 0.0095, 0.0095, 0.0095, 0.0095,
     . 0.0095, 0.0094, 0.0094, 0.0094, 0.0094, 0.0094, 0.0094, 0.0094,
     . 0.0094, 0.0094, 0.0094, 0.0094, 0.0094, 0.0096, 0.0097, 0.0097,
     . 0.0097, 0.0097, 0.0097, 0.0098, 0.0099, 0.0099, 0.0100, 0.0100,
     . 0.0100, 0.0101, 0.0101, 0.0101, 0.0101, 0.0101, 0.0101, 0.0102,
     . 0.0102, 0.0102, 0.0103, 0.0103, 0.0103, 0.0103, 0.0103, 0.0104,
     . 0.0081, 0.0081, 0.0081, 0.0081, 0.0081, 0.0080, 0.0081, 0.0081,
     . 0.0081, 0.0080, 0.0080, 0.0080, 0.0081, 0.0081, 0.0081, 0.0081,
     . 0.0082, 0.0082, 0.0081, 0.0080, 0.0080, 0.0080, 0.0080, 0.0080,
     . 0.0080, 0.0079, 0.0079, 0.0079, 0.0079, 0.0079, 0.0079, 0.0079,
     . 0.0079, 0.0079, 0.0079, 0.0079, 0.0079, 0.0081, 0.0082, 0.0082,
     . 0.0082, 0.0082, 0.0082, 0.0083, 0.0084, 0.0084, 0.0085, 0.0085,
     . 0.0085, 0.0086, 0.0086, 0.0086, 0.0086, 0.0086, 0.0086, 0.0087,
     . 0.0087, 0.0087, 0.0088, 0.0088, 0.0088, 0.0088, 0.0088, 0.0089,
     . 0.0066, 0.0066, 0.0066, 0.0066, 0.0066, 0.0066, 0.0066, 0.0066,
     . 0.0066, 0.0065, 0.0065, 0.0065, 0.0066, 0.0066, 0.0066, 0.0066,
     . 0.0066, 0.0067, 0.0066, 0.0065, 0.0065, 0.0065, 0.0065, 0.0065,
     . 0.0065, 0.0064, 0.0064, 0.0064, 0.0064, 0.0064, 0.0064, 0.0064,
     . 0.0064, 0.0064, 0.0064, 0.0064, 0.0065, 0.0066, 0.0067, 0.0067,
     . 0.0067, 0.0067, 0.0067, 0.0067, 0.0069, 0.0069, 0.0069, 0.0070,
     . 0.0070, 0.0070, 0.0071, 0.0071, 0.0071, 0.0071, 0.0071, 0.0072,
     . 0.0072, 0.0072, 0.0073, 0.0073, 0.0073, 0.0073, 0.0073, 0.0074,
     . 0.0052, 0.0051, 0.0051, 0.0051, 0.0051, 0.0051, 0.0051, 0.0051,
     . 0.0051, 0.0050, 0.0050, 0.0050, 0.0051, 0.0051, 0.0051, 0.0051,
     . 0.0051, 0.0052, 0.0051, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050,
     . 0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0049,
     . 0.0049, 0.0049, 0.0049, 0.0050, 0.0050, 0.0051, 0.0051, 0.0052,
     . 0.0052, 0.0052, 0.0052, 0.0052, 0.0054, 0.0054, 0.0054, 0.0055,
     . 0.0055, 0.0055, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0057,
     . 0.0057, 0.0057, 0.0058, 0.0058, 0.0058, 0.0058, 0.0058, 0.0059,
     . 0.0037, 0.0036, 0.0036, 0.0036, 0.0036, 0.0036, 0.0036, 0.0036,
     . 0.0036, 0.0036, 0.0035, 0.0035, 0.0036, 0.0036, 0.0036, 0.0036,
     . 0.0036, 0.0037, 0.0036, 0.0035, 0.0035, 0.0035, 0.0035, 0.0035,
     . 0.0035, 0.0035, 0.0035, 0.0035, 0.0035, 0.0035, 0.0035, 0.0034,
     . 0.0034, 0.0035, 0.0035, 0.0035, 0.0035, 0.0036, 0.0036, 0.0037,
     . 0.0037, 0.0037, 0.0037, 0.0037, 0.0039, 0.0039, 0.0039, 0.0040,
     . 0.0040, 0.0040, 0.0041, 0.0041, 0.0041, 0.0041, 0.0041, 0.0042,
     . 0.0042, 0.0042, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0044,
     . 0.0022, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021,
     . 0.0021, 0.0021, 0.0020, 0.0020, 0.0021, 0.0021, 0.0021, 0.0021,
     . 0.0022, 0.0022, 0.0021, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020,
     . 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020,
     . 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0021, 0.0021, 0.0021,
     . 0.0022, 0.0022, 0.0022, 0.0022, 0.0024, 0.0024, 0.0024, 0.0025,
     . 0.0025, 0.0025, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0027,
     . 0.0027, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0029,
     . 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007,
     . 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0008, 0.0007, 0.0007,
     . 0.0007, 0.0007, 0.0006, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
     . 0.0005, 0.0005, 0.0005, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006,
     . 0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0007, 0.0007, 0.0007,
     . 0.0007, 0.0008, 0.0008, 0.0008, 0.0010, 0.0010, 0.0011, 0.0011,
     . 0.0011, 0.0012, 0.0012, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013,
     . 0.0014, 0.0014, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0016,
     .-0.0008,-0.0008,-0.0008,-0.0008,-0.0008,-0.0008,-0.0008,-0.0008,
     .-0.0008,-0.0008,-0.0008,-0.0008,-0.0007,-0.0007,-0.0007,-0.0008,
     .-0.0008,-0.0008,-0.0009,-0.0010,-0.0010,-0.0010,-0.0010,-0.0010,
     .-0.0010,-0.0009,-0.0009,-0.0009,-0.0009,-0.0009,-0.0008,-0.0009,
     .-0.0009,-0.0009,-0.0009,-0.0009,-0.0009,-0.0008,-0.0008,-0.0008,
     .-0.0008,-0.0007,-0.0007,-0.0007,-0.0005,-0.0005,-0.0004,-0.0004,
     .-0.0003,-0.0003,-0.0002,-0.0002,-0.0002,-0.0002,-0.0002,-0.0001,
     .-0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0001, 0.0001,
     .-0.0023,-0.0023,-0.0023,-0.0023,-0.0023,-0.0023,-0.0022,-0.0022,
     .-0.0022,-0.0022,-0.0023,-0.0023,-0.0022,-0.0022,-0.0022,-0.0023,
     .-0.0023,-0.0023,-0.0024,-0.0025,-0.0025,-0.0025,-0.0025,-0.0025,
     .-0.0025,-0.0024,-0.0024,-0.0024,-0.0024,-0.0023,-0.0023,-0.0024,
     .-0.0024,-0.0024,-0.0024,-0.0023,-0.0023,-0.0023,-0.0023,-0.0023,
     .-0.0023,-0.0022,-0.0022,-0.0022,-0.0020,-0.0020,-0.0019,-0.0019,
     .-0.0018,-0.0018,-0.0017,-0.0017,-0.0017,-0.0017,-0.0016,-0.0016,
     .-0.0015,-0.0015,-0.0015,-0.0015,-0.0015,-0.0014,-0.0014,-0.0014,
     .-0.0038,-0.0038,-0.0038,-0.0038,-0.0038,-0.0038,-0.0037,-0.0037,
     .-0.0037,-0.0037,-0.0037,-0.0037,-0.0037,-0.0037,-0.0037,-0.0038,
     .-0.0038,-0.0038,-0.0039,-0.0040,-0.0040,-0.0040,-0.0040,-0.0040,
     .-0.0039,-0.0039,-0.0039,-0.0039,-0.0038,-0.0038,-0.0038,-0.0038,
     .-0.0038,-0.0038,-0.0038,-0.0038,-0.0038,-0.0038,-0.0038,-0.0038,
     .-0.0038,-0.0037,-0.0037,-0.0037,-0.0035,-0.0035,-0.0034,-0.0034,
     .-0.0033,-0.0033,-0.0032,-0.0032,-0.0032,-0.0031,-0.0031,-0.0031,
     .-0.0030,-0.0030,-0.0030,-0.0030,-0.0029,-0.0029,-0.0029,-0.0029,
     .-0.0052,-0.0052,-0.0052,-0.0052,-0.0052,-0.0052,-0.0052,-0.0052,
     .-0.0052,-0.0052,-0.0052,-0.0052,-0.0052,-0.0052,-0.0052,-0.0052,
     .-0.0053,-0.0053,-0.0054,-0.0055,-0.0055,-0.0055,-0.0055,-0.0054,
     .-0.0054,-0.0054,-0.0053,-0.0053,-0.0053,-0.0053,-0.0053,-0.0053,
     .-0.0053,-0.0053,-0.0053,-0.0053,-0.0053,-0.0053,-0.0053,-0.0053,
     .-0.0052,-0.0052,-0.0052,-0.0052,-0.0050,-0.0050,-0.0049,-0.0049,
     .-0.0048,-0.0048,-0.0047,-0.0047,-0.0047,-0.0046,-0.0046,-0.0046,
     .-0.0045,-0.0045,-0.0045,-0.0044,-0.0044,-0.0044,-0.0044,-0.0044,
     .-0.0067,-0.0067,-0.0067,-0.0067,-0.0067,-0.0067,-0.0067,-0.0067,
     .-0.0067,-0.0067,-0.0067,-0.0067,-0.0067,-0.0067,-0.0067,-0.0067,
     .-0.0068,-0.0068,-0.0068,-0.0070,-0.0070,-0.0069,-0.0069,-0.0069,
     .-0.0069,-0.0068,-0.0068,-0.0068,-0.0068,-0.0068,-0.0067,-0.0068,
     .-0.0068,-0.0068,-0.0068,-0.0068,-0.0068,-0.0068,-0.0068,-0.0068,
     .-0.0067,-0.0067,-0.0067,-0.0067,-0.0065,-0.0065,-0.0064,-0.0064,
     .-0.0063,-0.0063,-0.0062,-0.0062,-0.0061,-0.0061,-0.0061,-0.0061,
     .-0.0060,-0.0060,-0.0059,-0.0059,-0.0059,-0.0059,-0.0059,-0.0058,
     .-0.0081,-0.0081,-0.0081,-0.0081,-0.0081,-0.0081,-0.0081,-0.0081,
     .-0.0081,-0.0081,-0.0081,-0.0081,-0.0081,-0.0080,-0.0081,-0.0081,
     .-0.0081,-0.0081,-0.0082,-0.0083,-0.0083,-0.0083,-0.0083,-0.0083,
     .-0.0083,-0.0082,-0.0082,-0.0082,-0.0082,-0.0081,-0.0081,-0.0082,
     .-0.0082,-0.0082,-0.0082,-0.0081,-0.0081,-0.0082,-0.0082,-0.0082,
     .-0.0081,-0.0081,-0.0081,-0.0081,-0.0079,-0.0079,-0.0079,-0.0078,
     .-0.0078,-0.0078,-0.0077,-0.0076,-0.0076,-0.0076,-0.0076,-0.0075,
     .-0.0075,-0.0074,-0.0074,-0.0074,-0.0074,-0.0073,-0.0073,-0.0073,
     .-0.0095,-0.0095,-0.0095,-0.0095,-0.0095,-0.0095,-0.0095,-0.0095,
     .-0.0095,-0.0095,-0.0095,-0.0095,-0.0094,-0.0094,-0.0094,-0.0094,
     .-0.0095,-0.0095,-0.0095,-0.0096,-0.0096,-0.0096,-0.0096,-0.0096,
     .-0.0097,-0.0096,-0.0096,-0.0096,-0.0096,-0.0096,-0.0095,-0.0096,
     .-0.0096,-0.0096,-0.0096,-0.0095,-0.0095,-0.0095,-0.0095,-0.0095,
     .-0.0095,-0.0095,-0.0095,-0.0094,-0.0093,-0.0093,-0.0093,-0.0093,
     .-0.0093,-0.0092,-0.0091,-0.0091,-0.0091,-0.0091,-0.0090,-0.0090,
     .-0.0089,-0.0089,-0.0089,-0.0089,-0.0088,-0.0088,-0.0088,-0.0088,
     .-0.0110,-0.0110,-0.0110,-0.0110,-0.0110,-0.0110,-0.0110,-0.0109,
     .-0.0109,-0.0109,-0.0109,-0.0109,-0.0109,-0.0109,-0.0109,-0.0109,
     .-0.0109,-0.0110,-0.0110,-0.0111,-0.0111,-0.0111,-0.0111,-0.0111,
     .-0.0111,-0.0111,-0.0111,-0.0111,-0.0110,-0.0110,-0.0110,-0.0111,
     .-0.0111,-0.0110,-0.0110,-0.0110,-0.0110,-0.0110,-0.0110,-0.0110,
     .-0.0110,-0.0110,-0.0109,-0.0109,-0.0108,-0.0108,-0.0108,-0.0108,
     .-0.0107,-0.0107,-0.0106,-0.0106,-0.0106,-0.0106,-0.0105,-0.0105,
     .-0.0104,-0.0104,-0.0104,-0.0104,-0.0103,-0.0103,-0.0103,-0.0103,
     .-0.0125,-0.0125,-0.0125,-0.0125,-0.0125,-0.0125,-0.0124,-0.0124,
     .-0.0124,-0.0124,-0.0124,-0.0124,-0.0124,-0.0123,-0.0124,-0.0124,
     .-0.0124,-0.0124,-0.0125,-0.0126,-0.0126,-0.0126,-0.0126,-0.0126,
     .-0.0126,-0.0126,-0.0125,-0.0125,-0.0125,-0.0125,-0.0125,-0.0126,
     .-0.0125,-0.0125,-0.0125,-0.0125,-0.0125,-0.0125,-0.0125,-0.0125,
     .-0.0124,-0.0124,-0.0124,-0.0124,-0.0123,-0.0123,-0.0123,-0.0122,
     .-0.0122,-0.0122,-0.0121,-0.0121,-0.0121,-0.0120,-0.0120,-0.0120,
     .-0.0119,-0.0119,-0.0119,-0.0118,-0.0118,-0.0118,-0.0118,-0.0118,
     .-0.0140,-0.0140,-0.0140,-0.0140,-0.0140,-0.0140,-0.0139,-0.0139,
     .-0.0139,-0.0139,-0.0139,-0.0139,-0.0138,-0.0138,-0.0138,-0.0139,
     .-0.0139,-0.0139,-0.0139,-0.0140,-0.0140,-0.0140,-0.0141,-0.0141,
     .-0.0141,-0.0140,-0.0140,-0.0140,-0.0140,-0.0140,-0.0140,-0.0140,
     .-0.0140,-0.0140,-0.0140,-0.0140,-0.0140,-0.0139,-0.0139,-0.0139,
     .-0.0139,-0.0139,-0.0139,-0.0138,-0.0138,-0.0138,-0.0137,-0.0137,
     .-0.0137,-0.0137,-0.0136,-0.0136,-0.0136,-0.0135,-0.0135,-0.0135,
     .-0.0134,-0.0134,-0.0134,-0.0133,-0.0133,-0.0133,-0.0133,-0.0132,
     .-0.0155,-0.0155,-0.0155,-0.0154,-0.0154,-0.0154,-0.0154,-0.0154,
     .-0.0154,-0.0154,-0.0154,-0.0154,-0.0153,-0.0153,-0.0153,-0.0153,
     .-0.0153,-0.0154,-0.0154,-0.0155,-0.0155,-0.0155,-0.0155,-0.0155,
     .-0.0155,-0.0155,-0.0155,-0.0155,-0.0155,-0.0155,-0.0155,-0.0155,
     .-0.0155,-0.0155,-0.0155,-0.0155,-0.0154,-0.0154,-0.0154,-0.0154,
     .-0.0154,-0.0153,-0.0153,-0.0153,-0.0153,-0.0152,-0.0152,-0.0152,
     .-0.0152,-0.0152,-0.0151,-0.0151,-0.0150,-0.0150,-0.0150,-0.0150,
     .-0.0149,-0.0149,-0.0149,-0.0148,-0.0148,-0.0148,-0.0148,-0.0147,
     .-0.0169,-0.0169,-0.0169,-0.0169,-0.0169,-0.0169,-0.0169,-0.0168,
     .-0.0168,-0.0168,-0.0168,-0.0168,-0.0168,-0.0167,-0.0167,-0.0168,
     .-0.0168,-0.0168,-0.0168,-0.0169,-0.0169,-0.0169,-0.0169,-0.0170,
     .-0.0170,-0.0170,-0.0170,-0.0170,-0.0170,-0.0169,-0.0169,-0.0169,
     .-0.0169,-0.0169,-0.0169,-0.0169,-0.0169,-0.0168,-0.0168,-0.0168,
     .-0.0168,-0.0168,-0.0168,-0.0167,-0.0167,-0.0167,-0.0167,-0.0166,
     .-0.0166,-0.0166,-0.0165,-0.0165,-0.0165,-0.0165,-0.0165,-0.0164,
     .-0.0164,-0.0163,-0.0163,-0.0163,-0.0163,-0.0163,-0.0162,-0.0162,
     .-0.0184,-0.0184,-0.0184,-0.0183,-0.0183,-0.0183,-0.0183,-0.0183,
     .-0.0182,-0.0182,-0.0182,-0.0182,-0.0182,-0.0181,-0.0181,-0.0181,
     .-0.0180,-0.0180,-0.0180,-0.0181,-0.0181,-0.0181,-0.0181,-0.0181,
     .-0.0181,-0.0181,-0.0181,-0.0181,-0.0181,-0.0181,-0.0181,-0.0181,
     .-0.0181,-0.0181,-0.0181,-0.0180,-0.0180,-0.0180,-0.0179,-0.0179,
     .-0.0179,-0.0179,-0.0179,-0.0179,-0.0178,-0.0178,-0.0178,-0.0178,
     .-0.0178,-0.0178,-0.0177,-0.0177,-0.0177,-0.0177,-0.0177,-0.0177,
     .-0.0176,-0.0176,-0.0176,-0.0176,-0.0176,-0.0176,-0.0176,-0.0176,
     .-0.0199,-0.0198,-0.0198,-0.0198,-0.0198,-0.0198,-0.0197,-0.0197,
     .-0.0197,-0.0197,-0.0197,-0.0197,-0.0196,-0.0196,-0.0196,-0.0195,
     .-0.0195,-0.0195,-0.0195,-0.0196,-0.0196,-0.0196,-0.0196,-0.0196,
     .-0.0196,-0.0196,-0.0196,-0.0196,-0.0196,-0.0196,-0.0196,-0.0196,
     .-0.0196,-0.0196,-0.0195,-0.0195,-0.0195,-0.0195,-0.0194,-0.0194,
     .-0.0194,-0.0194,-0.0194,-0.0193,-0.0193,-0.0193,-0.0193,-0.0193,
     .-0.0192,-0.0192,-0.0192,-0.0192,-0.0192,-0.0192,-0.0192,-0.0191,
     .-0.0191,-0.0191,-0.0191,-0.0191,-0.0191,-0.0191,-0.0191,-0.0191,
     .-0.0213,-0.0213,-0.0213,-0.0213,-0.0213,-0.0213,-0.0212,-0.0212,
     .-0.0212,-0.0212,-0.0212,-0.0212,-0.0211,-0.0210,-0.0210,-0.0210,
     .-0.0210,-0.0210,-0.0210,-0.0211,-0.0210,-0.0210,-0.0210,-0.0210,
     .-0.0210,-0.0210,-0.0210,-0.0210,-0.0210,-0.0210,-0.0210,-0.0211,
     .-0.0211,-0.0210,-0.0210,-0.0210,-0.0210,-0.0209,-0.0209,-0.0209,
     .-0.0209,-0.0208,-0.0208,-0.0208,-0.0208,-0.0208,-0.0207,-0.0207,
     .-0.0207,-0.0207,-0.0207,-0.0207,-0.0207,-0.0206,-0.0206,-0.0206,
     .-0.0206,-0.0206,-0.0206,-0.0206,-0.0206,-0.0206,-0.0206,-0.0206,
     .-0.0228,-0.0228,-0.0228,-0.0228,-0.0228,-0.0227,-0.0227,-0.0227,
     .-0.0227,-0.0227,-0.0227,-0.0226,-0.0226,-0.0225,-0.0225,-0.0225,
     .-0.0225,-0.0225,-0.0225,-0.0225,-0.0225,-0.0225,-0.0225,-0.0225,
     .-0.0225,-0.0225,-0.0225,-0.0225,-0.0225,-0.0225,-0.0225,-0.0225,
     .-0.0225,-0.0225,-0.0225,-0.0225,-0.0225,-0.0224,-0.0223,-0.0223,
     .-0.0223,-0.0223,-0.0223,-0.0223,-0.0222,-0.0222,-0.0222,-0.0222,
     .-0.0222,-0.0222,-0.0222,-0.0222,-0.0221,-0.0221,-0.0221,-0.0221,
     .-0.0221,-0.0221,-0.0221,-0.0221,-0.0221,-0.0221,-0.0221,-0.0221,
     .-0.0243,-0.0243,-0.0243,-0.0242,-0.0242,-0.0242,-0.0242,-0.0241,
     .-0.0241,-0.0241,-0.0241,-0.0241,-0.0241,-0.0240,-0.0240,-0.0240,
     .-0.0240,-0.0240,-0.0239,-0.0240,-0.0240,-0.0240,-0.0240,-0.0240,
     .-0.0239,-0.0240,-0.0240,-0.0240,-0.0240,-0.0240,-0.0240,-0.0240,
     .-0.0240,-0.0240,-0.0240,-0.0240,-0.0239,-0.0239,-0.0238,-0.0238,
     .-0.0238,-0.0238,-0.0238,-0.0237,-0.0237,-0.0237,-0.0237,-0.0237,
     .-0.0236,-0.0236,-0.0236,-0.0236,-0.0236,-0.0236,-0.0236,-0.0236,
     .-0.0236,-0.0236,-0.0236,-0.0236,-0.0236,-0.0236,-0.0236,-0.0236,
     .-0.0257,-0.0257,-0.0257,-0.0257,-0.0257,-0.0257,-0.0256,-0.0256,
     .-0.0256,-0.0256,-0.0256,-0.0256,-0.0255,-0.0255,-0.0255,-0.0254,
     .-0.0254,-0.0254,-0.0254,-0.0254,-0.0254,-0.0254,-0.0254,-0.0254,
     .-0.0254,-0.0254,-0.0254,-0.0254,-0.0254,-0.0254,-0.0254,-0.0255,
     .-0.0255,-0.0255,-0.0254,-0.0254,-0.0254,-0.0253,-0.0253,-0.0253,
     .-0.0252,-0.0252,-0.0252,-0.0252,-0.0251,-0.0251,-0.0251,-0.0251,
     .-0.0251,-0.0251,-0.0251,-0.0251,-0.0251,-0.0251,-0.0251,-0.0251,
     .-0.0251,-0.0251,-0.0251,-0.0251,-0.0251,-0.0251,-0.0251,-0.0250,
     .-0.0273,-0.0273,-0.0273,-0.0273,-0.0273,-0.0273,-0.0272,-0.0272,
     .-0.0272,-0.0272,-0.0272,-0.0272,-0.0271,-0.0270,-0.0270,-0.0270,
     .-0.0270,-0.0270,-0.0270,-0.0270,-0.0269,-0.0269,-0.0269,-0.0269,
     .-0.0269,-0.0269,-0.0269,-0.0269,-0.0269,-0.0269,-0.0269,-0.0270,
     .-0.0269,-0.0269,-0.0269,-0.0269,-0.0269,-0.0268,-0.0268,-0.0268,
     .-0.0268,-0.0267,-0.0267,-0.0267,-0.0267,-0.0267,-0.0267,-0.0267,
     .-0.0267,-0.0267,-0.0267,-0.0267,-0.0267,-0.0267,-0.0267,-0.0267,
     .-0.0267,-0.0266,-0.0266,-0.0266,-0.0266,-0.0266,-0.0266,-0.0266,
     .-0.0288,-0.0288,-0.0288,-0.0288,-0.0288,-0.0287,-0.0287,-0.0287,
     .-0.0287,-0.0287,-0.0286,-0.0286,-0.0286,-0.0285,-0.0285,-0.0285,
     .-0.0284,-0.0284,-0.0284,-0.0284,-0.0284,-0.0284,-0.0284,-0.0284,
     .-0.0284,-0.0284,-0.0284,-0.0284,-0.0284,-0.0284,-0.0284,-0.0284,
     .-0.0284,-0.0284,-0.0284,-0.0284,-0.0284,-0.0283,-0.0282,-0.0282,
     .-0.0282,-0.0282,-0.0282,-0.0282,-0.0282,-0.0282,-0.0281,-0.0281,
     .-0.0281,-0.0281,-0.0282,-0.0282,-0.0282,-0.0282,-0.0281,-0.0281,
     .-0.0281,-0.0281,-0.0281,-0.0281,-0.0281,-0.0281,-0.0281,-0.0280,
     .-0.0303,-0.0303,-0.0302,-0.0302,-0.0302,-0.0302,-0.0302,-0.0301,
     .-0.0301,-0.0301,-0.0301,-0.0301,-0.0300,-0.0300,-0.0299,-0.0299,
     .-0.0299,-0.0299,-0.0299,-0.0299,-0.0299,-0.0299,-0.0299,-0.0298,
     .-0.0298,-0.0299,-0.0298,-0.0298,-0.0298,-0.0298,-0.0298,-0.0299,
     .-0.0299,-0.0299,-0.0298,-0.0298,-0.0298,-0.0298,-0.0297,-0.0297,
     .-0.0297,-0.0297,-0.0297,-0.0297,-0.0296,-0.0296,-0.0296,-0.0296,
     .-0.0296,-0.0296,-0.0297,-0.0296,-0.0296,-0.0296,-0.0296,-0.0296,
     .-0.0296,-0.0296,-0.0296,-0.0296,-0.0296,-0.0295,-0.0295,-0.0295,
     .-0.0317,-0.0317,-0.0317,-0.0317,-0.0317,-0.0317,-0.0316,-0.0316,
     .-0.0316,-0.0316,-0.0316,-0.0316,-0.0315,-0.0314,-0.0314,-0.0314,
     .-0.0314,-0.0314,-0.0313,-0.0313,-0.0313,-0.0313,-0.0313,-0.0313,
     .-0.0313,-0.0313,-0.0313,-0.0313,-0.0313,-0.0313,-0.0313,-0.0313,
     .-0.0313,-0.0313,-0.0313,-0.0313,-0.0313,-0.0312,-0.0312,-0.0312,
     .-0.0312,-0.0311,-0.0311,-0.0311,-0.0311,-0.0311,-0.0311,-0.0311,
     .-0.0311,-0.0311,-0.0311,-0.0311,-0.0311,-0.0311,-0.0311,-0.0311,
     .-0.0311,-0.0310,-0.0310,-0.0310,-0.0310,-0.0310,-0.0310,-0.0310,
     .-0.0332,-0.0332,-0.0332,-0.0332,-0.0332,-0.0331,-0.0331,-0.0331,
     .-0.0331,-0.0330,-0.0330,-0.0330,-0.0330,-0.0329,-0.0329,-0.0329,
     .-0.0328,-0.0328,-0.0328,-0.0328,-0.0328,-0.0328,-0.0328,-0.0328,
     .-0.0328,-0.0328,-0.0328,-0.0328,-0.0328,-0.0328,-0.0328,-0.0328,
     .-0.0328,-0.0328,-0.0328,-0.0328,-0.0327,-0.0327,-0.0326,-0.0326,
     .-0.0326,-0.0326,-0.0326,-0.0326,-0.0326,-0.0326,-0.0326,-0.0326,
     .-0.0326,-0.0326,-0.0326,-0.0326,-0.0326,-0.0326,-0.0326,-0.0326,
     .-0.0325,-0.0325,-0.0325,-0.0325,-0.0325,-0.0325,-0.0325,-0.0325,
     .-0.0346,-0.0346,-0.0346,-0.0346,-0.0346,-0.0346,-0.0345,-0.0345,
     .-0.0345,-0.0345,-0.0345,-0.0345,-0.0344,-0.0343,-0.0343,-0.0343,
     .-0.0343,-0.0343,-0.0343,-0.0343,-0.0343,-0.0343,-0.0342,-0.0342,
     .-0.0342,-0.0342,-0.0342,-0.0342,-0.0342,-0.0342,-0.0342,-0.0342,
     .-0.0342,-0.0342,-0.0342,-0.0342,-0.0342,-0.0341,-0.0341,-0.0341,
     .-0.0341,-0.0341,-0.0341,-0.0341,-0.0340,-0.0340,-0.0340,-0.0340,
     .-0.0340,-0.0340,-0.0341,-0.0340,-0.0340,-0.0340,-0.0340,-0.0340,
     .-0.0340,-0.0340,-0.0340,-0.0340,-0.0340,-0.0339,-0.0339,-0.0339,
     .-0.0361,-0.0361,-0.0361,-0.0361,-0.0361,-0.0361,-0.0360,-0.0360,
     .-0.0360,-0.0360,-0.0360,-0.0360,-0.0359,-0.0358,-0.0358,-0.0357,
     .-0.0357,-0.0357,-0.0357,-0.0357,-0.0357,-0.0357,-0.0356,-0.0356,
     .-0.0356,-0.0356,-0.0356,-0.0356,-0.0356,-0.0356,-0.0356,-0.0357,
     .-0.0356,-0.0356,-0.0356,-0.0356,-0.0356,-0.0355,-0.0355,-0.0355,
     .-0.0355,-0.0355,-0.0355,-0.0355,-0.0355,-0.0355,-0.0355,-0.0355,
     .-0.0355,-0.0355,-0.0355,-0.0355,-0.0355,-0.0355,-0.0355,-0.0355,
     .-0.0354,-0.0354,-0.0354,-0.0354,-0.0354,-0.0354,-0.0354,-0.0354,
     .-0.0376,-0.0376,-0.0375,-0.0375,-0.0375,-0.0375,-0.0375,-0.0375,
     .-0.0374,-0.0374,-0.0374,-0.0374,-0.0373,-0.0373,-0.0372,-0.0372,
     .-0.0372,-0.0371,-0.0371,-0.0371,-0.0371,-0.0371,-0.0371,-0.0371,
     .-0.0371,-0.0371,-0.0371,-0.0371,-0.0371,-0.0371,-0.0371,-0.0371,
     .-0.0371,-0.0371,-0.0371,-0.0370,-0.0370,-0.0370,-0.0370,-0.0370,
     .-0.0370,-0.0369,-0.0369,-0.0369,-0.0370,-0.0370,-0.0370,-0.0370,
     .-0.0370,-0.0370,-0.0370,-0.0370,-0.0369,-0.0369,-0.0369,-0.0369,
     .-0.0369,-0.0369,-0.0369,-0.0369,-0.0369,-0.0369,-0.0368,-0.0368,
     .-0.0390,-0.0390,-0.0390,-0.0390,-0.0390,-0.0390,-0.0389,-0.0389,
     .-0.0389,-0.0389,-0.0389,-0.0389,-0.0388,-0.0387,-0.0387,-0.0387,
     .-0.0386,-0.0386,-0.0386,-0.0386,-0.0386,-0.0386,-0.0386,-0.0386,
     .-0.0385,-0.0385,-0.0385,-0.0385,-0.0385,-0.0385,-0.0385,-0.0385,
     .-0.0385,-0.0385,-0.0385,-0.0385,-0.0385,-0.0384,-0.0384,-0.0384,
     .-0.0384,-0.0384,-0.0384,-0.0384,-0.0384,-0.0384,-0.0384,-0.0384,
     .-0.0384,-0.0384,-0.0384,-0.0384,-0.0384,-0.0384,-0.0384,-0.0384,
     .-0.0384,-0.0384,-0.0384,-0.0383,-0.0383,-0.0383,-0.0383,-0.0383,
     .-0.0405,-0.0405,-0.0405,-0.0405,-0.0404,-0.0404,-0.0404,-0.0404,
     .-0.0404,-0.0404,-0.0403,-0.0403,-0.0403,-0.0402,-0.0401,-0.0401,
     .-0.0401,-0.0401,-0.0400,-0.0401,-0.0401,-0.0400,-0.0400,-0.0400,
     .-0.0400,-0.0400,-0.0400,-0.0400,-0.0400,-0.0400,-0.0400,-0.0400,
     .-0.0400,-0.0400,-0.0399,-0.0399,-0.0399,-0.0399,-0.0399,-0.0399,
     .-0.0399,-0.0399,-0.0399,-0.0399,-0.0399,-0.0399,-0.0399,-0.0399,
     .-0.0399,-0.0399,-0.0399,-0.0399,-0.0399,-0.0399,-0.0399,-0.0398,
     .-0.0398,-0.0398,-0.0398,-0.0398,-0.0398,-0.0398,-0.0398,-0.0398,
     .-0.0419,-0.0419,-0.0419,-0.0419,-0.0419,-0.0419,-0.0418,-0.0418,
     .-0.0418,-0.0418,-0.0418,-0.0418,-0.0417,-0.0416,-0.0416,-0.0416,
     .-0.0416,-0.0415,-0.0415,-0.0415,-0.0415,-0.0415,-0.0415,-0.0415,
     .-0.0415,-0.0414,-0.0414,-0.0414,-0.0414,-0.0414,-0.0414,-0.0414,
     .-0.0414,-0.0414,-0.0414,-0.0414,-0.0414,-0.0414,-0.0413,-0.0413,
     .-0.0413,-0.0413,-0.0413,-0.0413,-0.0414,-0.0414,-0.0414,-0.0414,
     .-0.0414,-0.0414,-0.0414,-0.0413,-0.0413,-0.0413,-0.0413,-0.0413,
     .-0.0413,-0.0413,-0.0413,-0.0413,-0.0413,-0.0412,-0.0412,-0.0412,
     .-0.0434,-0.0434,-0.0434,-0.0434,-0.0434,-0.0433,-0.0433,-0.0433,
     .-0.0433,-0.0433,-0.0433,-0.0432,-0.0432,-0.0431,-0.0431,-0.0430,
     .-0.0430,-0.0430,-0.0430,-0.0430,-0.0430,-0.0430,-0.0429,-0.0429,
     .-0.0429,-0.0429,-0.0429,-0.0429,-0.0429,-0.0429,-0.0429,-0.0429,
     .-0.0429,-0.0428,-0.0428,-0.0428,-0.0428,-0.0428,-0.0428,-0.0428,
     .-0.0428,-0.0428,-0.0428,-0.0428,-0.0429,-0.0429,-0.0429,-0.0429,
     .-0.0429,-0.0429,-0.0428,-0.0428,-0.0428,-0.0428,-0.0428,-0.0428,
     .-0.0428,-0.0427,-0.0427,-0.0427,-0.0427,-0.0427,-0.0427,-0.0427,
     .-0.0448,-0.0448,-0.0448,-0.0448,-0.0448,-0.0448,-0.0448,-0.0447,
     .-0.0447,-0.0447,-0.0447,-0.0447,-0.0446,-0.0445,-0.0445,-0.0445,
     .-0.0445,-0.0444,-0.0444,-0.0444,-0.0444,-0.0444,-0.0444,-0.0444,
     .-0.0444,-0.0443,-0.0443,-0.0443,-0.0443,-0.0443,-0.0443,-0.0443,
     .-0.0443,-0.0443,-0.0443,-0.0443,-0.0443,-0.0443,-0.0443,-0.0443,
     .-0.0443,-0.0443,-0.0443,-0.0443,-0.0443,-0.0443,-0.0443,-0.0443,
     .-0.0443,-0.0443,-0.0443,-0.0443,-0.0443,-0.0442,-0.0442,-0.0442,
     .-0.0442,-0.0442,-0.0442,-0.0442,-0.0442,-0.0442,-0.0442,-0.0442,
     .-0.0463,-0.0463,-0.0463,-0.0463,-0.0463,-0.0462,-0.0462,-0.0462,
     .-0.0462,-0.0462,-0.0462,-0.0461,-0.0461,-0.0460,-0.0460,-0.0459,
     .-0.0459,-0.0459,-0.0459,-0.0459,-0.0459,-0.0459,-0.0459,-0.0459,
     .-0.0458,-0.0458,-0.0458,-0.0458,-0.0458,-0.0458,-0.0458,-0.0457,
     .-0.0457,-0.0457,-0.0457,-0.0457,-0.0457,-0.0457,-0.0457,-0.0457,
     .-0.0457,-0.0457,-0.0457,-0.0457,-0.0458,-0.0458,-0.0458,-0.0458,
     .-0.0458,-0.0458,-0.0457,-0.0457,-0.0457,-0.0457,-0.0457,-0.0457,
     .-0.0457,-0.0457,-0.0457,-0.0456,-0.0456,-0.0456,-0.0456,-0.0456/
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/GC/WFC3IR/wfc3ir_gc.f"
c****                                         
c*********************************************
                                              
 
c
cCiao, Rolly...
c
cI haven't documented much of the WFC3/IR stuff.
c
cGenerally, it appears to be more stable than WFC3/UVIS or ACS.
c
cA reasonable precision for the solution is 0.01 pixel.  I haven't
cfound a need to improve it yet.
c
cThe plate scale of my solution appears to be something like 0.12099"/pixel.
cI get this by comparing the achieved offsets in pixels against the
cengineering offsets from the POS-TARGs. My y-axis is aligned with the
cPOS-TARG axis to about 0.2 degree, but I can't say how that's related
cto the PA_APER in the image.  I haven't done that calculation for
cWFC3/IR yet.  That's waiting my sabbatical, I guess.
c
cIs this enough?
c  Jay
c
cPS: Congrats on getting M4 into the Kepler list.  That should be fun...
c
 
 
 
c---------------------------------------
c
c developed by Jay at home Xmas 2009
c
      subroutine wfc3ir_gcX(xr,yr,xc,yc)
      implicit none
 
      real*8 xr,yr
      real*8 xc,yc
 
      real xru,yru
 
      real x, y
      integer kinu
 
      real dxcor, dycor
 
      real polyx(10)
      common /IRpolyx_/polyx
      data polyx /
     .  507.000, 567.453,   1.767, -0.065,  7.171,
     .    0.033,  -0.002,   0.007,  0.003, -0.012/
 
      real polyy(10)
      common /IRpolyxyx_/polyy
      data polyy /
     .  507.000,   0.000, 507.000,  1.805, -0.049,
     .    7.866,   0.015,  -0.022,  0.032, -0.012/
 
      real xcor(11,11)
      common /IRxcor_/xcor
      data xcor /
     .0.1070,0.0915,0.0738,0.0580,0.0333,0.0178,
     .       0.0032,-.0196,-.0343,-.0387,-.0326,
     .0.0815,0.0691,0.0547,0.0387,0.0144,0.0040,
     .       -.0156,-.0389,-.0477,-.0479,-.0376,
     .0.0589,0.0498,0.0386,0.0244,0.0032,-.0044,
     .       -.0260,-.0462,-.0508,-.0468,-.0323,
     .0.0343,0.0328,0.0285,0.0243,0.0081,-.0089,
     .       -.0303,-.0391,-.0442,-.0346,-.0142,
     .0.0146,0.0084,0.0065,0.0175,0.0076,-.0127,
     .       -.0350,-.0348,-.0243,-.0165,-.0041,
     .0.0039,-.0020,-.0027,0.0071,0.0017,-.0049,
     .       -.0175,-.0209,-.0111,-.0005,0.0163,
     .-.0204,-.0184,-.0137,-.0036,-.0070,-.0009,
     .       -.0016,-.0085,-.0012,0.0149,0.0396,
     .-.0370,-.0347,-.0274,-.0081,-.0016,0.0032,
     .       0.0068,0.0141,0.0235,0.0401,0.0616,
     .-.0470,-.0455,-.0390,-.0224,-.0071,0.0043,
     .       0.0078,0.0223,0.0412,0.0543,0.0686,
     .-.0491,-.0480,-.0419,-.0229,-.0067,0.0074,
     .       0.0165,0.0343,0.0514,0.0656,0.0810,
     .-.0452,-.0444,-.0387,-.0196,-.0055,0.0127,
     .       0.0324,0.0497,0.0583,0.0736,0.0901/
 
      real ycor(11,11)
      common /IRycor_/ycor
      data ycor /
     .0.1085,0.0819,0.0633,0.0621,0.0369,0.0438,
     .       0.0580,0.0933,0.1052,0.1388,0.1840,
     .0.0703,0.0479,0.0336,0.0389,0.0233,0.0340,
     .       0.0343,0.0532,0.0650,0.0919,0.1305,
     .0.0360,0.0178,0.0077,0.0158,0.0050,0.0172,
     .       0.0097,0.0176,0.0252,0.0455,0.0773,
     .0.0100,-.0094,-.0184,-.0146,-.0162,-.0056,
     .       -.0190,-.0171,-.0240,-.0039,0.0324,
     .0.0013,-.0239,-.0373,-.0308,-.0363,-.0289,
     .       -.0336,-.0329,-.0470,-.0272,0.0105,
     .0.0193,-.0198,-.0428,-.0377,-.0466,-.0465,
     .       -.0521,-.0507,-.0556,-.0353,0.0008,
     .0.0367,-.0045,-.0324,-.0496,-.0577,-.0528,
     .       -.0624,-.0599,-.0466,-.0249,0.0115,
     .0.0634,0.0252,-.0033,-.0346,-.0489,-.0462,
     .       -.0542,-.0497,-.0351,-.0114,0.0275,
     .0.0930,0.0612,0.0357,0.0086,-.0063,-.0081,
     .       -.0110,-.0046,0.0092,0.0289,0.0584,
     .0.1118,0.0855,0.0653,0.0459,0.0339,0.0313,
     .       0.0298,0.0350,0.0435,0.0594,0.0851,
     .0.1303,0.1093,0.0945,0.0889,0.0813,0.0766,
     .       0.0771,0.0781,0.0802,0.0923,0.1143/
 
 
      real    rx, ry
      integer ix, iy
      real    fx, fy
 
      xc = xr
      yc = yr
 
      xru = xr
      yru = yr
 
      x = (xru-507.00)/507.0
      y = (yru-507.00)/507.0
      xc = polyx(01)
     .   + polyx(02)*x
     .   + polyx(03)*y
     .   + polyx(04)*x*x
     .   + polyx(05)*x*y
     .   + polyx(06)*y*y
     .   + polyx(07)*x*x*x
     .   + polyx(08)*x*x*y
     .   + polyx(09)*x*y*y
     .   + polyx(10)*y*y*y
      yc = polyy(01)
     .   + polyy(02)*x
     .   + polyy(03)*y
     .   + polyy(04)*x*x
     .   + polyy(05)*x*y
     .   + polyy(06)*y*y
     .   + polyy(07)*x*x*x
     .   + polyy(08)*x*x*y
     .   + polyy(09)*x*y*y
     .   + polyy(10)*y*y*y
 
 
      rx = 1 + xr/100
      ry = 1 + yr/100
      ix = int(rx)
      iy = int(ry)
      if (ix.lt.01) ix = 01
      if (iy.lt.01) iy = 01
      if (ix.gt.10) ix = 10
      if (iy.gt.10) iy = 10
      fx = rx-ix
      fy = ry-iy
 
      dxcor = (1-fx)*(1-fy)*xcor(ix  ,iy  )
     .      + (1-fx)*( fy )*xcor(ix  ,iy+1)
     .      + ( fx )*(1-fy)*xcor(ix+1,iy  )
     .      + ( fx )*( fy )*xcor(ix+1,iy+1)
 
      dycor = (1-fx)*(1-fy)*ycor(ix  ,iy  )
     .      + (1-fx)*( fy )*ycor(ix  ,iy+1)
     .      + ( fx )*(1-fy)*ycor(ix+1,iy  )
     .      + ( fx )*( fy )*ycor(ix+1,iy+1)
 
      xc = xc - dxcor
      yc = yc - dycor
 
      return
      end
 
 
